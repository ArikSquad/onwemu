//! Guest address space and the checked memory operations used by every
//! emulator subsystem.
//!
//! Mappings carry read/write and execute permissions, PSP RAM aliases resolve
//! to the same backing region, and successful writes bump 4 KiB page
//! generations for the CPU's self-modifying-code checks. The public methods
//! deliberately return structured faults so callers can report the guest PC
//! and thread that caused an invalid access.

use std::cell::Cell;
use std::collections::hash_map::{Entry, HashMap};
use std::fmt;
use thiserror::Error;

#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
/// A 32-bit address in the guest address space.
pub struct GuestAddress(pub u32);

impl fmt::Display for GuestAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0x{:08x}", self.0)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// The operation that failed a guest memory check.
pub enum AccessType {
    /// A guest read was rejected.
    Read,
    /// A guest write was rejected.
    Write,
    /// An instruction fetch was rejected.
    Execute,
}

#[derive(Debug, Error, Eq, PartialEq)]
#[error("{access:?} fault at {address} (pc={pc}, thread={thread_id}): {reason}")]
/// Details for a rejected guest memory access.
pub struct MemoryFault {
    /// Guest address at which the access failed.
    pub address: GuestAddress,
    /// Guest program counter recorded when the fault was created.
    pub pc: GuestAddress,
    /// Guest thread id recorded when the fault was created.
    pub thread_id: u32,
    /// Kind of access that was rejected.
    pub access: AccessType,
    /// Short stable explanation suitable for logs and diagnostics.
    pub reason: &'static str,
}

/// The checked byte-order and permission operations implemented by guest memory.
pub trait GuestMemory {
    /// Read one byte from guest memory.
    fn read_u8(&self, address: u32) -> Result<u8, MemoryFault>;
    /// Read a little-endian aligned halfword from guest memory.
    fn read_u16(&self, address: u32) -> Result<u16, MemoryFault>;
    /// Read a little-endian aligned word from guest memory.
    fn read_u32(&self, address: u32) -> Result<u32, MemoryFault>;
    /// Write one byte to guest memory.
    fn write_u8(&mut self, address: u32, value: u8) -> Result<(), MemoryFault>;
    /// Write a little-endian aligned halfword to guest memory.
    fn write_u16(&mut self, address: u32, value: u16) -> Result<(), MemoryFault>;
    /// Write a little-endian aligned word to guest memory.
    fn write_u32(&mut self, address: u32, value: u32) -> Result<(), MemoryFault>;
}

#[derive(Clone, Debug)]
struct Region {
    base: u32,
    data: Vec<u8>,
    writable: bool,
    executable: bool,
}

#[derive(Clone, Debug)]
/// Mapped guest memory with permission checks and PSP RAM alias handling.
pub struct Memory {
    regions: Vec<Region>,
    page_regions: HashMap<u32, usize>,
    /// Direct-mapped fast-path cache: page number to region lookup result.
    /// Hot instruction fetches and data accesses hit here instead of the
    /// `page_regions` hash map. Entries become stale only when a new mapping is
    /// created, which clears the whole cache.
    tlb: [Cell<TlbEntry>; TLB_ENTRIES],
    /// Per-4 KiB-page write generation, bumped on every guest write. The CPU
    /// block JIT snapshots these for the pages its compiled code lives on and
    /// recompiles when they change, which keeps self-modifying guest code
    /// correct without revalidating every block entry. A flat array keeps the
    /// hot validation lookup to a bounds check and load.
    page_generations: Vec<u32>,
    pc: u32,
    thread_id: u32,
}

/// Number of 4 KiB pages in the 32-bit address space.
const PAGE_COUNT: usize = 1 << (32 - PAGE_SHIFT);

impl Default for Memory {
    fn default() -> Self {
        Self {
            regions: Vec::new(),
            page_regions: HashMap::new(),
            tlb: [(); TLB_ENTRIES].map(|_| Cell::new(TlbEntry::default())),
            page_generations: vec![0; PAGE_COUNT],
            pc: 0,
            thread_id: 0,
        }
    }
}

const PAGE_SHIFT: u32 = 12;
const AMBIGUOUS_PAGE: usize = usize::MAX;
const TLB_ENTRIES: usize = 32;
const TLB_INVALID_PAGE: u32 = u32::MAX;

#[derive(Clone, Copy, Debug)]
struct TlbEntry {
    page: u32,
    region: usize,
    base: u32,
    end: u32,
    writable: bool,
    executable: bool,
}

impl Default for TlbEntry {
    fn default() -> Self {
        Self {
            page: TLB_INVALID_PAGE,
            region: 0,
            base: 1,
            end: 0,
            writable: false,
            executable: false,
        }
    }
}

/// Resolve the cached and uncached user/kernel views of the PSP's 32 MiB RAM to
/// the cached user view. Addresses outside physical RAM are left untouched; in
/// particular, arbitrary additions such as `0x11xx_xxxx` are not aliases.
#[inline(always)]
fn canonical_address(address: u32) -> u32 {
    let physical = address & 0x3fff_ffff;
    if (0x0800_0000..0x0a00_0000).contains(&physical) {
        physical
    } else {
        address
    }
}

impl Memory {
    /// Add a zero-filled guest mapping.
    ///
    /// `writable` and `executable` are checked by runtime writes and
    /// instruction fetches respectively. Mappings may not overlap.
    pub fn map(
        &mut self,
        base: u32,
        size: usize,
        writable: bool,
        executable: bool,
    ) -> Result<(), MemoryFault> {
        let base = canonical_address(base);
        let end = base
            .checked_add(
                u32::try_from(size)
                    .map_err(|_| self.fault(base, AccessType::Write, "region too large"))?,
            )
            .ok_or_else(|| self.fault(base, AccessType::Write, "region wraps address space"))?;
        if self
            .regions
            .iter()
            .any(|r| base < r.base + r.data.len() as u32 && r.base < end)
        {
            return Err(self.fault(base, AccessType::Write, "overlapping mapping"));
        }
        let region_index = self.regions.len();
        self.regions.push(Region {
            base,
            data: vec![0; size],
            writable,
            executable,
        });
        if size != 0 {
            let first_page = base >> PAGE_SHIFT;
            let last_page = (end - 1) >> PAGE_SHIFT;
            for page in first_page..=last_page {
                match self.page_regions.entry(page) {
                    Entry::Vacant(entry) => {
                        entry.insert(region_index);
                    }
                    Entry::Occupied(mut entry) => {
                        *entry.get_mut() = AMBIGUOUS_PAGE;
                    }
                }
            }
        }
        // region bases, sizes, and permissions never change after creation,
        // but a new mapping can shadow page->region associations, so the
        // fast-path cache must go.
        self.tlb_invalidate_all();
        Ok(())
    }

    /// Add a mapping and initialize its prefix from `initial` bytes.
    pub fn map_initialized(
        &mut self,
        base: u32,
        size: usize,
        initial: &[u8],
        writable: bool,
        executable: bool,
    ) -> Result<(), MemoryFault> {
        if initial.len() > size {
            return Err(self.fault(base, AccessType::Write, "initial data exceeds mapping"));
        }
        self.map(base, size, writable, executable)?;
        let region = self
            .regions
            .last_mut()
            .expect("map succeeded but did not create a region");
        region.data[..initial.len()].copy_from_slice(initial);
        Ok(())
    }

    #[inline(always)]
    /// Attach a guest PC and thread id to faults produced by later accesses.
    pub fn set_context(&mut self, pc: u32, thread_id: u32) {
        self.pc = pc;
        self.thread_id = thread_id;
    }

    #[inline(always)]
    /// Write a checked byte range and bump all touched page generations.
    pub fn write_bytes(&mut self, address: u32, bytes: &[u8]) -> Result<(), MemoryFault> {
        let mapped_address = canonical_address(address);
        let page = mapped_address >> PAGE_SHIFT;
        if let Some(entry) = self.tlb_probe(page)
            && (mapped_address as u64).saturating_add(bytes.len() as u64) <= entry.end as u64
            && mapped_address >= entry.base
        {
            if !entry.writable {
                return Err(MemoryFault {
                    address: GuestAddress(address),
                    pc: GuestAddress(self.pc),
                    thread_id: self.thread_id,
                    access: AccessType::Write,
                    reason: "read-only mapping",
                });
            }
            let offset = (mapped_address - entry.base) as usize;
            self.regions[entry.region].data[offset..offset + bytes.len()].copy_from_slice(bytes);
            self.bump_generations(mapped_address, bytes.len());
            return Ok(());
        }
        let pc = self.pc;
        let thread_id = self.thread_id;
        let r = self.region_mut(mapped_address, bytes.len(), AccessType::Write)?;
        if !r.writable {
            return Err(MemoryFault {
                address: GuestAddress(address),
                pc: GuestAddress(pc),
                thread_id,
                access: AccessType::Write,
                reason: "read-only mapping",
            });
        }
        let offset = (mapped_address - r.base) as usize;
        r.data[offset..offset + bytes.len()].copy_from_slice(bytes);
        self.bump_generations(mapped_address, bytes.len());
        self.tlb_insert(page);
        Ok(())
    }

    /// Fill a guest range without allocating a temporary host buffer.
    ///
    /// This is the primitive used by CPU fast paths for byte-at-a-time memset
    /// loops emitted by PSP toolchains.
    #[inline(always)]
    pub fn fill_bytes(
        &mut self,
        address: u32,
        length: usize,
        value: u8,
    ) -> Result<(), MemoryFault> {
        let mapped_address = canonical_address(address);
        let page = mapped_address >> PAGE_SHIFT;
        if let Some(entry) = self.tlb_probe(page)
            && (mapped_address as u64).saturating_add(length as u64) <= entry.end as u64
            && mapped_address >= entry.base
        {
            if !entry.writable {
                return Err(MemoryFault {
                    address: GuestAddress(address),
                    pc: GuestAddress(self.pc),
                    thread_id: self.thread_id,
                    access: AccessType::Write,
                    reason: "read-only mapping",
                });
            }
            let offset = (mapped_address - entry.base) as usize;
            self.regions[entry.region].data[offset..offset + length].fill(value);
            self.bump_generations(mapped_address, length);
            return Ok(());
        }
        let pc = self.pc;
        let thread_id = self.thread_id;
        let r = self.region_mut(mapped_address, length, AccessType::Write)?;
        if !r.writable {
            return Err(MemoryFault {
                address: GuestAddress(address),
                pc: GuestAddress(pc),
                thread_id,
                access: AccessType::Write,
                reason: "read-only mapping",
            });
        }
        let offset = (mapped_address - r.base) as usize;
        r.data[offset..offset + length].fill(value);
        self.bump_generations(mapped_address, length);
        self.tlb_insert(page);
        Ok(())
    }

    /// Copy a checked guest range into a host-owned vector.
    pub fn read_bytes(&self, address: u32, length: usize) -> Result<Vec<u8>, MemoryFault> {
        Ok(self.read_slice(address, length)?.to_vec())
    }

    /// Borrow a checked guest range for bulk reads without a host allocation.
    pub fn read_slice(&self, address: u32, length: usize) -> Result<&[u8], MemoryFault> {
        let mapped_address = canonical_address(address);
        let page = mapped_address >> PAGE_SHIFT;
        if let Some(entry) = self.tlb_probe(page)
            && (mapped_address as u64).saturating_add(length as u64) <= entry.end as u64
            && mapped_address >= entry.base
        {
            let offset = (mapped_address - entry.base) as usize;
            return Ok(&self.regions[entry.region].data[offset..offset + length]);
        }
        let r = self.region(mapped_address, length, AccessType::Read)?;
        let offset = (mapped_address - r.base) as usize;
        let bytes = &r.data[offset..offset + length];
        self.tlb_insert(page);
        Ok(bytes)
    }

    /// Return the current write generation for a 4 KiB page.
    ///
    /// The CPU block JIT uses this value to detect when compiled guest code
    /// changes. A page that has never been written has generation zero.
    #[inline(always)]
    pub fn page_generation(&self, page: u32) -> u32 {
        self.page_generations[(canonical_address(page << PAGE_SHIFT) >> PAGE_SHIFT) as usize]
    }

    #[inline(always)]
    fn tlb_probe(&self, page: u32) -> Option<TlbEntry> {
        let entry = self.tlb[(page as usize) & (TLB_ENTRIES - 1)].get();
        (entry.page == page).then_some(entry)
    }

    /// Cache the region covering `page`'s start address. Callers must have
    /// resolved the access through [`Self::region`] or [`Self::region_mut`]
    /// first; a failed access is never cached.
    fn tlb_insert(&self, page: u32) {
        let address = page << PAGE_SHIFT;
        let Some(index) = self.find_region_index(address, address.wrapping_add(1)) else {
            return;
        };
        let region = &self.regions[index];
        let end = region.base.wrapping_add(region.data.len() as u32);
        self.tlb[(page as usize) & (TLB_ENTRIES - 1)].set(TlbEntry {
            page,
            region: index,
            base: region.base,
            end,
            writable: region.writable,
            executable: region.executable,
        });
    }

    fn tlb_invalidate_all(&self) {
        for slot in &self.tlb {
            slot.set(TlbEntry::default());
        }
    }

    /// Record a successful guest write so JIT-compiled blocks covering the
    /// touched pages recompile on the next entry. Call this only after the
    /// write has been bounds-checked and applied.
    fn bump_generations(&mut self, mapped_address: u32, length: usize) {
        if length == 0 {
            return;
        }
        let first = mapped_address >> PAGE_SHIFT;
        let last = mapped_address.wrapping_add(length as u32 - 1) >> PAGE_SHIFT;
        for page in first..=last {
            let slot = &mut self.page_generations
                [(canonical_address(page << PAGE_SHIFT) >> PAGE_SHIFT) as usize];
            *slot = slot.wrapping_add(1);
        }
    }

    /// Write a word while constructing an image, before guest permissions
    /// apply.
    ///
    /// This is intentionally separate from [`GuestMemory`] so runtime code
    /// cannot use it to bypass read-only mappings.
    pub fn patch_u32(&mut self, address: u32, value: u32) -> Result<(), MemoryFault> {
        if address & 3 != 0 {
            return Err(self.fault(address, AccessType::Write, "unaligned loader patch"));
        }
        let mapped_address = canonical_address(address);
        let region = self.region_mut(mapped_address, 4, AccessType::Write)?;
        let offset = (mapped_address - region.base) as usize;
        region.data[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        self.bump_generations(mapped_address, 4);
        Ok(())
    }

    #[inline(always)]
    /// Fetch an aligned executable word in guest little-endian order.
    pub fn fetch_u32(&self, address: u32) -> Result<u32, MemoryFault> {
        if address & 3 != 0 {
            return Err(self.fault(address, AccessType::Execute, "unaligned instruction fetch"));
        }
        let mapped_address = canonical_address(address);
        // an aligned word never straddles a page boundary, so a single-page
        // probe is sufficient here.
        let page = mapped_address >> PAGE_SHIFT;
        if let Some(entry) = self.tlb_probe(page)
            && mapped_address >= entry.base
            && mapped_address.wrapping_add(4) <= entry.end
        {
            if !entry.executable {
                return Err(self.fault(address, AccessType::Execute, "non-executable mapping"));
            }
            let offset = (mapped_address - entry.base) as usize;
            return Ok(u32::from_le_bytes(
                self.regions[entry.region].data[offset..offset + 4]
                    .try_into()
                    .unwrap(),
            ));
        }
        let end = mapped_address
            .checked_add(4)
            .ok_or_else(|| self.fault(address, AccessType::Execute, "access wraps"))?;
        let Some(index) = self.find_region_index(mapped_address, end) else {
            return Err(self.fault(address, AccessType::Execute, "unmapped address"));
        };
        let region = &self.regions[index];
        if !region.executable {
            return Err(self.fault(address, AccessType::Execute, "non-executable mapping"));
        }
        let offset = (mapped_address - region.base) as usize;
        let word = u32::from_le_bytes(region.data[offset..offset + 4].try_into().unwrap());
        self.tlb_insert(page);
        Ok(word)
    }

    #[inline(always)]
    fn fault(&self, address: u32, access: AccessType, reason: &'static str) -> MemoryFault {
        MemoryFault {
            address: GuestAddress(address),
            pc: GuestAddress(self.pc),
            thread_id: self.thread_id,
            access,
            reason,
        }
    }
    #[inline(always)]
    fn region(&self, address: u32, len: usize, access: AccessType) -> Result<&Region, MemoryFault> {
        let end = address
            .checked_add(
                u32::try_from(len).map_err(|_| self.fault(address, access, "access too large"))?,
            )
            .ok_or_else(|| self.fault(address, access, "access wraps"))?;
        self.find_region_index(address, end)
            .and_then(|index| self.regions.get(index))
            .ok_or_else(|| self.fault(address, access, "unmapped address"))
    }
    #[inline(always)]
    fn region_mut(
        &mut self,
        address: u32,
        len: usize,
        access: AccessType,
    ) -> Result<&mut Region, MemoryFault> {
        let end = address
            .checked_add(
                u32::try_from(len).map_err(|_| self.fault(address, access, "access too large"))?,
            )
            .ok_or_else(|| self.fault(address, access, "access wraps"))?;
        let Some(index) = self.find_region_index(address, end) else {
            return Err(self.fault(address, access, "unmapped address"));
        };
        Ok(&mut self.regions[index])
    }

    #[inline(always)]
    fn find_region_index(&self, address: u32, end: u32) -> Option<usize> {
        match self.page_regions.get(&(address >> PAGE_SHIFT)) {
            Some(&index) if index != AMBIGUOUS_PAGE => {
                let region = &self.regions[index];
                (address >= region.base && end <= region.base + region.data.len() as u32)
                    .then_some(index)
            }
            Some(_) | None => self.regions.iter().position(|region| {
                address >= region.base && end <= region.base + region.data.len() as u32
            }),
        }
    }
}

impl GuestMemory for Memory {
    #[inline(always)]
    fn read_u8(&self, address: u32) -> Result<u8, MemoryFault> {
        let mapped_address = canonical_address(address);
        let page = mapped_address >> PAGE_SHIFT;
        if let Some(entry) = self.tlb_probe(page)
            && mapped_address >= entry.base
            && mapped_address.wrapping_add(1) <= entry.end
        {
            return Ok(self.regions[entry.region].data[(mapped_address - entry.base) as usize]);
        }
        let r = self.region(mapped_address, 1, AccessType::Read)?;
        let value = r.data[(mapped_address - r.base) as usize];
        self.tlb_insert(page);
        Ok(value)
    }
    #[inline(always)]
    fn read_u16(&self, address: u32) -> Result<u16, MemoryFault> {
        if address & 1 != 0 {
            return Err(self.fault(address, AccessType::Read, "unaligned halfword read"));
        }
        let mapped_address = canonical_address(address);
        let page = mapped_address >> PAGE_SHIFT;
        if let Some(entry) = self.tlb_probe(page)
            && mapped_address >= entry.base
            && mapped_address.wrapping_add(2) <= entry.end
        {
            let o = (mapped_address - entry.base) as usize;
            return Ok(u16::from_le_bytes(
                self.regions[entry.region].data[o..o + 2]
                    .try_into()
                    .unwrap(),
            ));
        }
        let r = self.region(mapped_address, 2, AccessType::Read)?;
        let o = (mapped_address - r.base) as usize;
        let value = u16::from_le_bytes(r.data[o..o + 2].try_into().unwrap());
        self.tlb_insert(page);
        Ok(value)
    }
    #[inline(always)]
    fn read_u32(&self, address: u32) -> Result<u32, MemoryFault> {
        if address & 3 != 0 {
            return Err(self.fault(address, AccessType::Read, "unaligned word read"));
        }
        let mapped_address = canonical_address(address);
        let page = mapped_address >> PAGE_SHIFT;
        if let Some(entry) = self.tlb_probe(page)
            && mapped_address >= entry.base
            && mapped_address.wrapping_add(4) <= entry.end
        {
            let o = (mapped_address - entry.base) as usize;
            return Ok(u32::from_le_bytes(
                self.regions[entry.region].data[o..o + 4]
                    .try_into()
                    .unwrap(),
            ));
        }
        let r = self.region(mapped_address, 4, AccessType::Read)?;
        let o = (mapped_address - r.base) as usize;
        let value = u32::from_le_bytes(r.data[o..o + 4].try_into().unwrap());
        self.tlb_insert(page);
        Ok(value)
    }
    #[inline(always)]
    fn write_u32(&mut self, address: u32, value: u32) -> Result<(), MemoryFault> {
        if address & 3 != 0 {
            return Err(self.fault(address, AccessType::Write, "unaligned word write"));
        }
        self.write_bytes(address, &value.to_le_bytes())
    }
    #[inline(always)]
    fn write_u8(&mut self, address: u32, value: u8) -> Result<(), MemoryFault> {
        self.write_bytes(address, &[value])
    }
    #[inline(always)]
    fn write_u16(&mut self, address: u32, value: u16) -> Result<(), MemoryFault> {
        if address & 1 != 0 {
            return Err(self.fault(address, AccessType::Write, "unaligned halfword write"));
        }
        self.write_bytes(address, &value.to_le_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn little_endian_and_bounds() {
        let mut m = Memory::default();
        m.map(0x1000, 8, true, true).unwrap();
        m.write_u32(0x1000, 0x12345678).unwrap();
        assert_eq!(m.read_u8(0x1000).unwrap(), 0x78);
        assert_eq!(m.read_bytes(0x1000, 4).unwrap(), [0x78, 0x56, 0x34, 0x12]);
        assert!(m.read_u32(0x1006).is_err());
    }

    #[test]
    fn fill_bytes_writes_without_a_temporary_buffer() {
        let mut m = Memory::default();
        m.map(0x1000, 8, true, true).unwrap();
        m.fill_bytes(0x1002, 4, 0xa5).unwrap();
        assert_eq!(
            m.read_bytes(0x1000, 8).unwrap(),
            [0, 0, 0xa5, 0xa5, 0xa5, 0xa5, 0, 0]
        );
    }
    #[test]
    fn bulk_reads_reject_ranges_that_wrap_or_cross_mapping_boundaries() {
        let mut m = Memory::default();
        m.map(0x1000, 4, true, false).unwrap();
        m.write_u32(0x1000, 0x1234_5678).unwrap();
        assert_eq!(m.read_slice(0x1000, 4).unwrap(), &[0x78, 0x56, 0x34, 0x12]);
        assert!(m.read_slice(0x1000, 5).is_err());
        assert!(m.read_slice(0x1000, usize::MAX).is_err());
        assert!(m.read_slice(0xffff_fffc, 8).is_err());
    }

    #[test]
    fn permissions() {
        let mut m = Memory::default();
        m.map(0, 4, false, true).unwrap();
        assert!(m.write_u32(0, 1).is_err());
    }
    #[test]
    fn psp_ram_aliases_share_backing_memory() {
        let mut m = Memory::default();
        m.map(0x0800_0000, 4, true, true).unwrap();
        m.write_u32(0x0800_0000, 0x1234_5678).unwrap();
        assert_eq!(m.read_u32(0x4800_0000).unwrap(), 0x1234_5678);
        assert_eq!(m.read_u32(0x8800_0000).unwrap(), 0x1234_5678);
        m.write_u32(0xc800_0000, 0x89ab_cdef).unwrap();
        assert_eq!(m.read_u32(0x0800_0000).unwrap(), 0x89ab_cdef);
    }
    #[test]
    fn aliases_and_loader_patches_share_write_generations() {
        let mut m = Memory::default();
        m.map(0x0800_0000, 8192, true, true).unwrap();
        m.write_bytes(0x8800_0ffe, &[1, 2, 3, 4]).unwrap();
        for base in [0x0800_0000, 0x4800_0000, 0x8800_0000, 0xc800_0000] {
            assert_eq!(m.page_generation(base >> 12), 1);
            assert_eq!(m.page_generation((base >> 12) + 1), 1);
        }
        m.patch_u32(0xc800_0000, 0).unwrap();
        assert_eq!(m.page_generation(0x0800_0000 >> 12), 2);
    }

    #[test]
    fn double_ram_base_is_not_an_alias() {
        let mut m = Memory::default();
        m.map(0x09e9_5eb0, 4, true, false).unwrap();
        assert!(m.write_u32(0x11e9_5eb0, 1).is_err());
    }

    #[test]
    fn page_index_preserves_unaligned_region_boundaries() {
        let mut m = Memory::default();
        m.map(0x6000, 0x100, true, false).unwrap();
        m.map(0x6200, 0x100, true, false).unwrap();
        m.write_u8(0x60ff, 0x11).unwrap();
        m.write_u8(0x6200, 0x22).unwrap();
        assert_eq!(m.read_u8(0x60ff).unwrap(), 0x11);
        assert_eq!(m.read_u8(0x6200).unwrap(), 0x22);
        assert!(m.read_u16(0x60ff).is_err());
    }

    #[test]
    fn faults_preserve_access_kind_context_and_reason() {
        let mut memory = Memory::default();
        memory.map(0x1000, 8, false, false).unwrap();
        memory.set_context(0x2222, 17);

        let read = memory.read_u16(0x1001).unwrap_err();
        assert_eq!(read.access, AccessType::Read);
        assert_eq!(read.pc, GuestAddress(0x2222));
        assert_eq!(read.thread_id, 17);
        assert_eq!(read.reason, "unaligned halfword read");

        let execute = memory.fetch_u32(0x1000).unwrap_err();
        assert_eq!(execute.access, AccessType::Execute);
        assert_eq!(execute.reason, "non-executable mapping");

        let write = memory.write_u8(0x1000, 1).unwrap_err();
        assert_eq!(write.access, AccessType::Write);
        assert_eq!(write.reason, "read-only mapping");
    }

    #[test]
    fn mappings_reject_overlap_wrap_and_oversized_initial_data() {
        let mut memory = Memory::default();
        memory.map(0x2000, 0x100, true, false).unwrap();
        assert!(memory.map(0x2080, 0x100, true, false).is_err());
        assert!(memory.map(u32::MAX - 3, 8, true, false).is_err());
        assert!(
            memory
                .map_initialized(0x3000, 2, &[1, 2, 3], true, false)
                .is_err()
        );
    }

    #[test]
    fn loader_patches_can_prepare_read_only_executable_code() {
        let mut memory = Memory::default();
        memory.map(0x4000, 4, false, true).unwrap();
        memory.patch_u32(0x4000, 0x1234_5678).unwrap();
        assert_eq!(memory.fetch_u32(0x4000).unwrap(), 0x1234_5678);
        assert!(memory.patch_u32(0x4001, 0).is_err());
        assert!(memory.write_u32(0x4000, 0).is_err());
    }
}
