//! PSP image inspection, ISO access, ELF mapping, and PRX import linking.
//!
//! The loader accepts only the executable formats and decryption paths that are
//! implemented locally. It never supplies firmware, keys, or Sony assets; a
//! retail PRX that needs unavailable keys returns an explicit error instead of
//! pretending the bytes were loaded successfully.

use psp_memory::{GuestMemory, Memory};
use serde::Serialize;
use std::{
    fs,
    io::{BufReader, Cursor},
    io::{Read, Seek, SeekFrom},
    path::Path,
};
use thiserror::Error;

mod prx;

#[derive(Debug, Error)]
/// Errors produced while inspecting, reading, or mapping a PSP image.
pub enum LoaderError {
    #[error("I/O error: {0}")]
    /// The host could not read the image.
    Io(#[from] std::io::Error),
    #[error("truncated {0}")]
    /// The image ended before a required structure was complete.
    Truncated(&'static str),
    #[error("unsupported format: {0}")]
    /// The image uses a format this loader does not implement.
    Unsupported(&'static str),
    #[error("invalid image: {0}")]
    /// The image has a malformed or inconsistent structure.
    Invalid(&'static str),
    #[error("invalid CHD image: {0}")]
    /// The CHD library rejected the image.
    Chd(String),
    #[error("ISO path not found: {0}")]
    /// An ISO lookup did not find the requested path.
    NotFound(String),
    #[error(
        "encrypted PSP executable {module} requires keys/firmware not supplied by this project"
    )]
    /// The image is encrypted with keys intentionally not shipped here.
    Encrypted {
        /// Module name reported by the image.
        module: String,
    },
    #[error("PSP executable decryption failed: {0}")]
    /// A supported decryption path failed while processing the image.
    Decryption(String),
}
#[derive(Clone, Debug, Serialize)]
/// One loadable ELF segment and the guest range it occupies.
pub struct Segment {
    /// File offset of the segment's initialized bytes.
    pub offset: u32,
    /// Guest virtual address where the segment is mapped.
    pub address: u32,
    /// Number of bytes copied from the image.
    pub file_size: u32,
    /// Number of bytes reserved in guest memory, including BSS.
    pub memory_size: u32,
    /// ELF segment flags.
    pub flags: u32,
}

/// One firmware function referenced by a loaded PRX module.
///
/// `syscall` is the emulator-private number installed in the module's
/// two-word stub after linking.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Import {
    /// Emulator-private syscall number written into the import stub.
    pub syscall: u32,
    /// Firmware library named by the module.
    pub library: String,
    /// Firmware function identifier.
    pub nid: u32,
    /// Guest address of the two-word import stub.
    pub stub_address: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
/// The result of applying the loader's PRX import linking pass.
pub struct LinkedPrx {
    /// Global-pointer value recovered from the module.
    pub gp: u32,
    /// Imports found and linked in the module.
    pub imports: Vec<Import>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
/// A file or directory entry in an ISO-9660 image.
pub struct IsoEntry {
    /// Case-preserved ISO directory name.
    pub name: String,
    /// 2048-byte sector containing the entry.
    pub extent: u32,
    /// File length in bytes.
    pub size: u32,
    /// Whether the entry is a directory.
    pub directory: bool,
}

/// A read-only ISO-9660 view. All reads are bounded by the image length.
trait ReadSeek: Read + Seek {}
impl<T: Read + Seek> ReadSeek for T {}

const RAW_READ_AHEAD: usize = 256 * 1024;

struct RawReadCache {
    offset: u64,
    bytes: Vec<u8>,
}

/// An opened ISO-9660 or CHD image with bounded directory and raw reads.
pub struct IsoImage {
    file: Box<dyn ReadSeek>,
    image_len: u64,
    root: IsoEntry,
    raw_cache: Option<RawReadCache>,
}

impl IsoImage {
    /// Open an ISO or CHD and validate its primary volume descriptor.
    pub fn open(path: &Path) -> Result<Self, LoaderError> {
        let (file, image_len) = open_disc(path)?;
        Self::from_reader(file, image_len)
    }

    /// Open an ISO or CHD already held in memory and validate its volume
    /// descriptor. This is used by browser frontends that receive user-owned
    /// images through the File API.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, LoaderError> {
        let (file, image_len) = open_disc_bytes(bytes)?;
        Self::from_reader(file, image_len)
    }

    fn from_reader(mut file: Box<dyn ReadSeek>, image_len: u64) -> Result<Self, LoaderError> {
        file.seek(SeekFrom::Start(0x8000))?;
        let mut pvd = [0; 2048];
        file.read_exact(&mut pvd)?;
        if pvd[0] != 1 || &pvd[1..6] != b"CD001" {
            return Err(LoaderError::Invalid("primary volume descriptor"));
        }
        let root = parse_directory_record(&pvd[156..])?;
        Ok(Self {
            file,
            image_len,
            root,
            raw_cache: None,
        })
    }

    /// List the entries in an ISO directory.
    pub fn entries(&mut self, path: &str) -> Result<Vec<IsoEntry>, LoaderError> {
        let entry = self.find(path)?;
        if !entry.directory {
            return Err(LoaderError::Invalid("ISO path is not a directory"));
        }
        let bytes = self.read_entry(&entry, 64 * 1024 * 1024)?;
        directory_entries(&bytes)
    }

    /// Read a file while enforcing a caller-provided byte limit.
    pub fn read_file(&mut self, path: &str, limit: usize) -> Result<Vec<u8>, LoaderError> {
        let entry = self.find(path)?;
        if entry.directory {
            return Err(LoaderError::Invalid("ISO path is a directory"));
        }
        self.read_entry(&entry, limit)
    }

    /// Read an exact raw byte range from the underlying disc image.
    pub fn read_raw(&mut self, offset: u64, size: usize) -> Result<Vec<u8>, LoaderError> {
        self.validate_raw_range(offset, size as u64)?;
        let end = offset
            .checked_add(size as u64)
            .ok_or(LoaderError::Invalid("raw disc read overflow"))?;
        if let Some(cache) = &self.raw_cache {
            let cache_end = cache
                .offset
                .checked_add(cache.bytes.len() as u64)
                .ok_or(LoaderError::Invalid("raw disc cache overflow"))?;
            if offset >= cache.offset && end <= cache_end {
                let start = (offset - cache.offset) as usize;
                return Ok(cache.bytes[start..start + size].to_vec());
            }
        }

        // streaming loaders issue many adjacent 32 kib reads. a small
        // read-ahead turns those into one chd seek/decode while preserving
        // exact psp read semantics at the api boundary. random seeks simply
        // replace the cache with a new window.
        let available = usize::try_from(self.image_len - offset)
            .map_err(|_| LoaderError::Invalid("raw disc range exceeds host size"))?;
        let read_size = size.max(RAW_READ_AHEAD).min(available);
        self.file.seek(SeekFrom::Start(offset))?;
        let mut bytes = vec![0; read_size];
        self.file.read_exact(&mut bytes)?;
        let output = bytes[..size].to_vec();
        self.raw_cache = Some(RawReadCache { offset, bytes });
        Ok(output)
    }

    /// Check that a raw range is inside the image without reading it.
    pub fn validate_raw_range(&self, offset: u64, size: u64) -> Result<(), LoaderError> {
        let end = offset
            .checked_add(size)
            .ok_or(LoaderError::Invalid("raw disc read overflow"))?;
        if end > self.image_len {
            return Err(LoaderError::Truncated("raw disc extent"));
        }
        Ok(())
    }

    /// Find a case-insensitive path relative to the ISO root.
    pub fn find(&mut self, path: &str) -> Result<IsoEntry, LoaderError> {
        let mut current = self.root.clone();
        for wanted in path.split('/').filter(|part| !part.is_empty()) {
            if wanted == "." || wanted == ".." {
                return Err(LoaderError::Invalid("unsafe ISO path component"));
            }
            let bytes = self.read_entry(&current, 64 * 1024 * 1024)?;
            current = directory_entries(&bytes)?
                .into_iter()
                .find(|entry| entry.name.eq_ignore_ascii_case(wanted))
                .ok_or_else(|| LoaderError::NotFound(path.to_owned()))?;
        }
        Ok(current)
    }

    fn read_entry(&mut self, entry: &IsoEntry, limit: usize) -> Result<Vec<u8>, LoaderError> {
        let size =
            usize::try_from(entry.size).map_err(|_| LoaderError::Invalid("file too large"))?;
        if size > limit {
            return Err(LoaderError::Invalid("file exceeds configured read limit"));
        }
        let offset = u64::from(entry.extent)
            .checked_mul(2048)
            .ok_or(LoaderError::Invalid("ISO extent overflow"))?;
        let end = offset
            .checked_add(u64::from(entry.size))
            .ok_or(LoaderError::Invalid("ISO file overflow"))?;
        if end > self.image_len {
            return Err(LoaderError::Truncated("ISO file extent"));
        }
        self.file.seek(SeekFrom::Start(offset))?;
        let mut bytes = vec![0; size];
        self.file.read_exact(&mut bytes)?;
        Ok(bytes)
    }
}

fn open_disc(path: &Path) -> Result<(Box<dyn ReadSeek>, u64), LoaderError> {
    let mut file = fs::File::open(path)?;
    let mut magic = [0; 8];
    file.read_exact(&mut magic)?;
    file.seek(SeekFrom::Start(0))?;
    if &magic == b"MComprHD" {
        let chd = chd::Chd::open(BufReader::new(file), None)
            .map_err(|error| LoaderError::Chd(error.to_string()))?;
        let image_len = chd.header().logical_bytes();
        return Ok((Box::new(chd::read::ChdReader::new(chd)), image_len));
    }
    let image_len = file.metadata()?.len();
    Ok((Box::new(file), image_len))
}

fn open_disc_bytes(bytes: Vec<u8>) -> Result<(Box<dyn ReadSeek>, u64), LoaderError> {
    if bytes.starts_with(b"MComprHD") {
        let chd = chd::Chd::open(BufReader::new(Cursor::new(bytes)), None)
            .map_err(|error| LoaderError::Chd(error.to_string()))?;
        let image_len = chd.header().logical_bytes();
        return Ok((Box::new(chd::read::ChdReader::new(chd)), image_len));
    }
    let image_len = bytes.len() as u64;
    Ok((Box::new(Cursor::new(bytes)), image_len))
}
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type")]
/// Basic metadata returned by image inspection.
pub enum ImageInfo {
    /// An ELF image and its loadable segments.
    Elf {
        /// Entry point requested by the ELF header.
        entry: u32,
        /// Machine identifier from the ELF header.
        machine: u16,
        /// Loadable segments described by the image.
        segments: Vec<Segment>,
    },
    /// A PBP container and the metadata of its embedded executable, if any.
    Pbp {
        /// Offsets of the eight PBP sections.
        offsets: [u32; 8],
        /// Embedded executable metadata, when the PBP contains one.
        embedded_exec: Option<Box<ImageInfo>>,
    },
    /// An ISO-9660 image and its root directory summary.
    Iso {
        /// Volume identifier from the primary volume descriptor.
        volume_id: String,
        /// Root directory sector.
        root_extent: u32,
        /// Root directory byte length.
        root_size: u32,
        /// Names found directly under the root directory.
        root_entries: Vec<String>,
    },
}
fn u16le(b: &[u8], o: usize) -> Result<u16, LoaderError> {
    Ok(u16::from_le_bytes(
        b.get(o..o + 2)
            .ok_or(LoaderError::Truncated("u16"))?
            .try_into()
            .unwrap(),
    ))
}
fn u32le(b: &[u8], o: usize) -> Result<u32, LoaderError> {
    Ok(u32::from_le_bytes(
        b.get(o..o + 4)
            .ok_or(LoaderError::Truncated("u32"))?
            .try_into()
            .unwrap(),
    ))
}

/// Inspect an ELF, PBP, ISO, or CHD path without mapping it into guest memory.
pub fn inspect_path(path: &Path) -> Result<ImageInfo, LoaderError> {
    let mut raw = fs::File::open(path)?;
    let mut magic = [0; 8];
    raw.read_exact(&mut magic)?;
    raw.seek(SeekFrom::Start(0))?;
    if &magic != b"MComprHD" {
        let mut header = Vec::with_capacity(0x8800);
        raw.by_ref().take(0x8800).read_to_end(&mut header)?;
        if header.get(0x8001..0x8006) == Some(b"CD001") {
            return iso_reader(&mut raw, &header);
        }
        return inspect(&fs::read(path)?);
    }

    let (mut file, _) = open_disc(path)?;
    let mut header = Vec::with_capacity(0x8800);
    file.by_ref().take(0x8800).read_to_end(&mut header)?;
    if header.get(0x8001..0x8006) == Some(b"CD001") {
        return iso_reader(&mut file, &header);
    }
    Err(LoaderError::Invalid(
        "CHD does not contain an ISO 9660 image",
    ))
}
/// Inspect image bytes that are already available in memory.
pub fn inspect(b: &[u8]) -> Result<ImageInfo, LoaderError> {
    if b.starts_with(b"\x7fELF") {
        return elf(b);
    }
    if b.starts_with(b"\0PBP") {
        return pbp(b);
    }
    if b.get(0x8001..0x8006) == Some(b"CD001") {
        return iso(b);
    }
    Err(LoaderError::Unsupported("expected ELF, PBP, or ISO 9660"))
}
fn elf(b: &[u8]) -> Result<ImageInfo, LoaderError> {
    if b.get(4) != Some(&1) || b.get(5) != Some(&1) {
        return Err(LoaderError::Unsupported("ELF must be 32-bit little-endian"));
    }
    let machine = u16le(b, 18)?;
    if machine != 8 {
        return Err(LoaderError::Unsupported("ELF machine is not MIPS"));
    }
    let entry = u32le(b, 24)?;
    let phoff = u32le(b, 28)? as usize;
    let entsz = u16le(b, 42)? as usize;
    let count = u16le(b, 44)? as usize;
    if entsz < 32 {
        return Err(LoaderError::Invalid("ELF program header too small"));
    }
    let mut segments = Vec::new();
    for i in 0..count {
        let o = phoff
            .checked_add(
                i.checked_mul(entsz)
                    .ok_or(LoaderError::Invalid("header overflow"))?,
            )
            .ok_or(LoaderError::Invalid("header overflow"))?;
        if u32le(b, o)? == 1 {
            let s = Segment {
                offset: u32le(b, o + 4)?,
                address: u32le(b, o + 8)?,
                file_size: u32le(b, o + 16)?,
                memory_size: u32le(b, o + 20)?,
                flags: u32le(b, o + 24)?,
            };
            let end = s
                .offset
                .checked_add(s.file_size)
                .ok_or(LoaderError::Invalid("segment wraps"))? as usize;
            if end > b.len() || s.file_size > s.memory_size {
                return Err(LoaderError::Invalid("bad load segment"));
            }
            segments.push(s)
        }
    }
    Ok(ImageInfo::Elf {
        entry,
        machine,
        segments,
    })
}
fn pbp(b: &[u8]) -> Result<ImageInfo, LoaderError> {
    let mut offsets = [0; 8];
    for (i, x) in offsets.iter_mut().enumerate() {
        *x = u32le(b, 8 + i * 4)?
    }
    if !offsets.windows(2).all(|w| w[0] <= w[1]) || offsets[7] as usize > b.len() {
        return Err(LoaderError::Invalid("PBP offsets"));
    }
    let start = offsets[6] as usize;
    let end = offsets[7] as usize;
    let embedded_exec = if start < end {
        inspect(&b[start..end]).ok().map(Box::new)
    } else {
        None
    };
    Ok(ImageInfo::Pbp {
        offsets,
        embedded_exec,
    })
}
fn iso(b: &[u8]) -> Result<ImageInfo, LoaderError> {
    let pvd = b
        .get(0x8000..0x8800)
        .ok_or(LoaderError::Truncated("primary volume descriptor"))?;
    if pvd[0] != 1 || &pvd[1..6] != b"CD001" {
        return Err(LoaderError::Invalid("primary volume descriptor"));
    }
    let volume_id = String::from_utf8_lossy(&pvd[40..72]).trim().to_owned();
    let root = &pvd[156..];
    let root_extent = u32le(root, 2)?;
    let root_size = u32le(root, 10)?;
    let start = (root_extent as usize)
        .checked_mul(2048)
        .ok_or(LoaderError::Invalid("root extent overflow"))?;
    let end = start
        .checked_add(root_size as usize)
        .ok_or(LoaderError::Invalid("root size overflow"))?;
    let dir = b
        .get(start..end)
        .ok_or(LoaderError::Truncated("root directory"))?;
    let mut entries = Vec::new();
    let mut o = 0;
    while o < dir.len() {
        let len = dir[o] as usize;
        if len == 0 {
            o = ((o / 2048) + 1) * 2048;
            continue;
        }
        let rec = dir
            .get(o..o + len)
            .ok_or(LoaderError::Truncated("directory record"))?;
        let n = *rec
            .get(32)
            .ok_or(LoaderError::Truncated("directory name"))? as usize;
        let name = rec
            .get(33..33 + n)
            .ok_or(LoaderError::Truncated("directory name"))?;
        if name != [0] && name != [1] {
            entries.push(
                String::from_utf8_lossy(name)
                    .trim_end_matches(";1")
                    .to_owned(),
            )
        }
        o += len
    }
    Ok(ImageInfo::Iso {
        volume_id,
        root_extent,
        root_size,
        root_entries: entries,
    })
}

fn iso_reader(file: &mut dyn ReadSeek, header: &[u8]) -> Result<ImageInfo, LoaderError> {
    let pvd = header
        .get(0x8000..0x8800)
        .ok_or(LoaderError::Truncated("primary volume descriptor"))?;
    if pvd[0] != 1 || &pvd[1..6] != b"CD001" {
        return Err(LoaderError::Invalid("primary volume descriptor"));
    }
    let volume_id = String::from_utf8_lossy(&pvd[40..72]).trim().to_owned();
    let root = &pvd[156..];
    let root_extent = u32le(root, 2)?;
    let root_size = u32le(root, 10)?;
    file.seek(SeekFrom::Start(u64::from(root_extent) * 2048))?;
    let mut directory = vec![0; root_size as usize];
    file.read_exact(&mut directory)?;
    let root_entries = directory_names(&directory)?;
    Ok(ImageInfo::Iso {
        volume_id,
        root_extent,
        root_size,
        root_entries,
    })
}

fn directory_names(dir: &[u8]) -> Result<Vec<String>, LoaderError> {
    let mut entries = Vec::new();
    let mut o = 0;
    while o < dir.len() {
        let len = dir[o] as usize;
        if len == 0 {
            o = ((o / 2048) + 1) * 2048;
            continue;
        }
        let rec = dir
            .get(o..o + len)
            .ok_or(LoaderError::Truncated("directory record"))?;
        let n = *rec
            .get(32)
            .ok_or(LoaderError::Truncated("directory name"))? as usize;
        let name = rec
            .get(33..33 + n)
            .ok_or(LoaderError::Truncated("directory name"))?;
        if name != [0] && name != [1] {
            entries.push(
                String::from_utf8_lossy(name)
                    .trim_end_matches(";1")
                    .to_owned(),
            );
        }
        o += len;
    }
    Ok(entries)
}

fn parse_directory_record(rec: &[u8]) -> Result<IsoEntry, LoaderError> {
    let len = *rec
        .first()
        .ok_or(LoaderError::Truncated("directory record"))? as usize;
    let rec = rec
        .get(..len)
        .ok_or(LoaderError::Truncated("directory record"))?;
    let name_len = *rec
        .get(32)
        .ok_or(LoaderError::Truncated("directory name"))? as usize;
    let raw_name = rec
        .get(33..33 + name_len)
        .ok_or(LoaderError::Truncated("directory name"))?;
    let name = match raw_name {
        [0] => ".".to_owned(),
        [1] => "..".to_owned(),
        _ => String::from_utf8_lossy(raw_name)
            .trim_end_matches(";1")
            .to_owned(),
    };
    Ok(IsoEntry {
        name,
        extent: u32le(rec, 2)?,
        size: u32le(rec, 10)?,
        directory: rec.get(25).is_some_and(|flags| flags & 2 != 0),
    })
}

fn directory_entries(bytes: &[u8]) -> Result<Vec<IsoEntry>, LoaderError> {
    let mut result = Vec::new();
    let mut offset = 0;
    while offset < bytes.len() {
        let len = bytes[offset] as usize;
        if len == 0 {
            offset = ((offset / 2048) + 1) * 2048;
            continue;
        }
        let entry = parse_directory_record(&bytes[offset..])?;
        if entry.name != "." && entry.name != ".." {
            result.push(entry);
        }
        offset = offset
            .checked_add(len)
            .ok_or(LoaderError::Invalid("directory offset overflow"))?;
    }
    Ok(result)
}

/// Return a runnable ELF from a standalone ELF/PBP or a PSP ISO.
pub fn load_executable(path: &Path) -> Result<Vec<u8>, LoaderError> {
    let mut header = [0; 6];
    let mut file = fs::File::open(path)?;
    file.read_exact(&mut header)?;
    if &header[..4] == b"\x7fELF" {
        return Ok(fs::read(path)?);
    }
    if &header[..4] == b"\0PBP" {
        let bytes = fs::read(path)?;
        let info = pbp(&bytes)?;
        if let ImageInfo::Pbp { offsets, .. } = info {
            return executable_bytes(&bytes[offsets[6] as usize..offsets[7] as usize]);
        }
    }
    let mut iso = IsoImage::open(path)?;
    load_executable_from_disc(&mut iso)
}

/// Return a runnable ELF from an ELF, PBP, ISO, or CHD image held in memory.
pub fn load_executable_from_bytes(bytes: &[u8]) -> Result<Vec<u8>, LoaderError> {
    if bytes.starts_with(b"\x7fELF") {
        return executable_bytes(bytes);
    }
    if bytes.starts_with(b"\0PBP") {
        let info = pbp(bytes)?;
        if let ImageInfo::Pbp { offsets, .. } = info {
            return executable_bytes(&bytes[offsets[6] as usize..offsets[7] as usize]);
        }
    }
    let mut iso = IsoImage::from_bytes(bytes.to_vec())?;
    load_executable_from_disc(&mut iso)
}

/// Load the boot executable from an already opened PSP disc image.
pub fn load_executable_from_disc(iso: &mut IsoImage) -> Result<Vec<u8>, LoaderError> {
    let eboot = iso.read_file("PSP_GAME/SYSDIR/EBOOT.BIN", 64 * 1024 * 1024)?;
    executable_bytes(&eboot)
}

fn executable_bytes(bytes: &[u8]) -> Result<Vec<u8>, LoaderError> {
    if bytes.starts_with(b"\x7fELF") {
        elf(bytes)?;
        return Ok(bytes.to_vec());
    }
    if bytes.starts_with(b"~PSP") {
        let module = bytes
            .get(10..38)
            .map(|name| {
                String::from_utf8_lossy(name)
                    .trim_end_matches('\0')
                    .to_owned()
            })
            .unwrap_or_else(|| "unknown".to_owned());
        let decrypted = prx::decrypt(bytes).map_err(|error| match error {
            prx::PrxError::UnsupportedTag(_) => LoaderError::Encrypted { module },
            error => LoaderError::Decryption(error.to_string()),
        })?;
        elf(&decrypted)?;
        return Ok(decrypted);
    }
    Err(LoaderError::Unsupported("boot payload is not an ELF"))
}

/// Validate and map ELF load segments, returning the guest entry point.
pub fn map_elf(bytes: &[u8], memory: &mut Memory) -> Result<u32, LoaderError> {
    let info = elf(bytes)?;
    let ImageInfo::Elf {
        entry, segments, ..
    } = info
    else {
        return Err(LoaderError::Invalid("ELF parser returned wrong image type"));
    };
    if segments.is_empty() {
        return Err(LoaderError::Invalid("ELF has no loadable segments"));
    }
    // ET_SCE_PRX images use module-relative virtual addresses. The PSP loader
    // normally places a user module in the 0x0880_0000 partition and applies
    // its SHT_PSP_REL records before starting it.
    const ET_SCE_PRX: u16 = 0xffa0;
    const USER_MODULE_BASE: u32 = 0x0880_4000;
    let elf_type = u16le(bytes, 16)?;
    let load_base = if elf_type == ET_SCE_PRX {
        USER_MODULE_BASE
    } else {
        0
    };
    for segment in &segments {
        let start = segment.offset as usize;
        let end = start
            .checked_add(segment.file_size as usize)
            .ok_or(LoaderError::Invalid("segment slice overflow"))?;
        memory
            .map_initialized(
                segment
                    .address
                    .checked_add(load_base)
                    .ok_or(LoaderError::Invalid("segment address overflow"))?,
                segment.memory_size as usize,
                &bytes[start..end],
                // Allegrex user RAM has no per-page ELF permission enforcement.
                // Retail modules commonly update tables that linkers place in
                // a nominally read/execute segment.
                true,
                true,
            )
            .map_err(|_| LoaderError::Invalid("overlapping or invalid ELF mapping"))?;
    }
    if elf_type == ET_SCE_PRX {
        apply_prx_relocations(bytes, memory, &segments, load_base)?;
    }
    entry
        .checked_add(load_base)
        .ok_or(LoaderError::Invalid("entry address overflow"))
}

/// Parse a PSP module's resident import tables and replace each firmware stub
/// with `jr $ra; syscall N`.
///
/// The original library/NID identity is returned for HLE dispatch. Call this
/// after [`map_elf`], which relocates table pointers in a PRX.
pub fn link_prx_imports(bytes: &[u8], memory: &mut Memory) -> Result<LinkedPrx, LoaderError> {
    const ET_SCE_PRX: u16 = 0xffa0;
    const USER_MODULE_BASE: u32 = 0x0880_4000;
    if u16le(bytes, 16)? != ET_SCE_PRX {
        return Ok(LinkedPrx {
            gp: 0,
            imports: Vec::new(),
        });
    }
    let ImageInfo::Elf { .. } = elf(bytes)? else {
        return Err(LoaderError::Invalid("ELF parser returned wrong image type"));
    };
    let phoff = u32le(bytes, 28)? as usize;
    let phentsize = u16le(bytes, 42)? as usize;
    let phnum = u16le(bytes, 44)? as usize;
    let mut module_info = None;
    for index in 0..phnum {
        let header = phoff
            .checked_add(
                index
                    .checked_mul(phentsize)
                    .ok_or(LoaderError::Invalid("header overflow"))?,
            )
            .ok_or(LoaderError::Invalid("header overflow"))?;
        if u32le(bytes, header)? != 1 {
            continue;
        }
        let file_offset = u32le(bytes, header + 4)?;
        let physical = u32le(bytes, header + 12)? & 0x3fff_ffff;
        let file_size = u32le(bytes, header + 16)?;
        if physical >= file_offset && physical < file_offset.saturating_add(file_size) {
            module_info = Some(
                USER_MODULE_BASE
                    .checked_add(u32le(bytes, header + 8)?)
                    .and_then(|address| address.checked_add(physical - file_offset))
                    .ok_or(LoaderError::Invalid("module info address overflow"))?,
            );
            break;
        }
    }
    let module_info = module_info.ok_or(LoaderError::Invalid("PRX module info is not mapped"))?;
    let gp = memory
        .read_u32(module_info + 32)
        .map_err(|_| LoaderError::Invalid("bad PRX global pointer"))?;
    let stub_top = memory
        .read_u32(module_info + 44)
        .map_err(|_| LoaderError::Invalid("bad PRX stub top"))?;
    let stub_end = memory
        .read_u32(module_info + 48)
        .map_err(|_| LoaderError::Invalid("bad PRX stub end"))?;
    if stub_top > stub_end || stub_end - stub_top > 1024 * 1024 {
        return Err(LoaderError::Invalid("bad PRX import table range"));
    }
    let mut imports = Vec::new();
    let mut descriptor = stub_top;
    while descriptor < stub_end {
        let name_address = memory
            .read_u32(descriptor)
            .map_err(|_| LoaderError::Invalid("bad PRX import descriptor"))?;
        let size_words = memory
            .read_u8(descriptor + 8)
            .map_err(|_| LoaderError::Invalid("bad PRX import descriptor"))?
            as u32;
        let function_count = memory
            .read_u16(descriptor + 10)
            .map_err(|_| LoaderError::Invalid("bad PRX import descriptor"))?
            as u32;
        let nid_table = memory
            .read_u32(descriptor + 12)
            .map_err(|_| LoaderError::Invalid("bad PRX NID table"))?;
        let stub_table = memory
            .read_u32(descriptor + 16)
            .map_err(|_| LoaderError::Invalid("bad PRX stub table"))?;
        if size_words < 5
            || descriptor
                .checked_add(size_words * 4)
                .is_none_or(|next| next > stub_end)
        {
            return Err(LoaderError::Invalid("bad PRX import descriptor size"));
        }
        let mut name = Vec::new();
        for offset in 0..128u32 {
            let byte = memory
                .read_u8(name_address + offset)
                .map_err(|_| LoaderError::Invalid("bad PRX import library name"))?;
            if byte == 0 {
                break;
            }
            name.push(byte);
        }
        if name.len() == 128 {
            return Err(LoaderError::Invalid("unterminated PRX import library name"));
        }
        let library = String::from_utf8(name)
            .map_err(|_| LoaderError::Invalid("non-UTF-8 PRX import library name"))?;
        for index in 0..function_count {
            let syscall = u32::try_from(imports.len() + 1)
                .map_err(|_| LoaderError::Invalid("too many PRX imports"))?;
            if syscall > 0x000f_ffff {
                return Err(LoaderError::Invalid("too many PRX imports"));
            }
            let nid = memory
                .read_u32(nid_table + index * 4)
                .map_err(|_| LoaderError::Invalid("bad PRX import NID"))?;
            let stub_address = stub_table
                .checked_add(index * 8)
                .ok_or(LoaderError::Invalid("PRX stub address overflow"))?;
            memory
                .patch_u32(stub_address, 0x03e0_0008)
                .map_err(|_| LoaderError::Invalid("cannot patch PRX import return"))?;
            memory
                .patch_u32(stub_address + 4, (syscall << 6) | 0x0c)
                .map_err(|_| LoaderError::Invalid("cannot patch PRX import syscall"))?;
            imports.push(Import {
                syscall,
                library: library.clone(),
                nid,
                stub_address,
            });
        }
        descriptor += size_words * 4;
    }
    Ok(LinkedPrx { gp, imports })
}

const SHT_PSP_REL: u32 = 0x7000_00a0;

fn apply_prx_relocations(
    bytes: &[u8],
    memory: &mut Memory,
    segments: &[Segment],
    load_base: u32,
) -> Result<(), LoaderError> {
    let shoff = u32le(bytes, 32)? as usize;
    let shentsize = u16le(bytes, 46)? as usize;
    let shnum = u16le(bytes, 48)? as usize;
    if shentsize < 40 {
        return Err(LoaderError::Invalid("ELF section header too small"));
    }
    for section_index in 0..shnum {
        let header = shoff
            .checked_add(
                section_index
                    .checked_mul(shentsize)
                    .ok_or(LoaderError::Invalid("section header overflow"))?,
            )
            .ok_or(LoaderError::Invalid("section header overflow"))?;
        if u32le(bytes, header + 4)? != SHT_PSP_REL {
            continue;
        }
        let offset = u32le(bytes, header + 16)? as usize;
        let size = u32le(bytes, header + 20)? as usize;
        if !size.is_multiple_of(8) || offset.checked_add(size).is_none_or(|end| end > bytes.len()) {
            return Err(LoaderError::Invalid("bad PSP relocation section"));
        }
        let mut pending_hi = Vec::<(u32, u32, u8)>::new();
        for rel in (offset..offset + size).step_by(8) {
            let rel_offset = u32le(bytes, rel)?;
            let info = u32le(bytes, rel + 4)?;
            let kind = info as u8;
            let offset_segment = ((info >> 8) & 0xff) as usize;
            let value_segment = ((info >> 16) & 0xff) as usize;
            let location = segments
                .get(offset_segment)
                .and_then(|segment| segment.address.checked_add(load_base))
                .and_then(|base| base.checked_add(rel_offset))
                .ok_or(LoaderError::Invalid("relocation location overflow"))?;
            let value_base = segments
                .get(value_segment)
                .and_then(|segment| segment.address.checked_add(load_base))
                .ok_or(LoaderError::Invalid("bad relocation segment"))?;
            let word = memory
                .read_u32(location)
                .map_err(|_| LoaderError::Invalid("relocation points outside mapped image"))?;
            match kind {
                2 => memory.patch_u32(location, word.wrapping_add(value_base)),
                4 => {
                    let target = (word & 0x03ff_ffff).wrapping_add(value_base >> 2) & 0x03ff_ffff;
                    memory.patch_u32(location, (word & 0xfc00_0000) | target)
                }
                5 => {
                    pending_hi.push((location, word, value_segment as u8));
                    Ok(())
                }
                6 => {
                    let low = word as u16 as i16 as i32 as u32;
                    for (hi_location, hi_word, _segment) in pending_hi
                        .iter()
                        .copied()
                        .filter(|entry| entry.2 == value_segment as u8)
                    {
                        let value = (hi_word << 16).wrapping_add(low).wrapping_add(value_base);
                        let patched = (hi_word & 0xffff_0000) | (value.wrapping_add(0x8000) >> 16);
                        memory
                            .patch_u32(hi_location, patched)
                            .map_err(|_| LoaderError::Invalid("cannot patch HI16 relocation"))?;
                    }
                    pending_hi.retain(|entry| entry.2 != value_segment as u8);
                    memory.patch_u32(
                        location,
                        (word & 0xffff_0000) | word.wrapping_add(value_base) & 0xffff,
                    )
                }
                _ => return Err(LoaderError::Unsupported("unknown PSP relocation type")),
            }
            .map_err(|_| LoaderError::Invalid("cannot patch PSP relocation"))?;
        }
        if !pending_hi.is_empty() {
            return Err(LoaderError::Invalid("unpaired HI16 relocation"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use psp_memory::GuestMemory;
    use std::cell::Cell;
    use std::io::{Cursor, Read, Seek, SeekFrom};
    use std::rc::Rc;

    fn synthetic_elf() -> Vec<u8> {
        let mut bytes = vec![0; 0x108];
        bytes[..4].copy_from_slice(b"\x7fELF");
        bytes[4] = 1;
        bytes[5] = 1;
        bytes[16..18].copy_from_slice(&2u16.to_le_bytes());
        bytes[18..20].copy_from_slice(&8u16.to_le_bytes());
        bytes[24..28].copy_from_slice(&0x0880_0000u32.to_le_bytes());
        bytes[28..32].copy_from_slice(&52u32.to_le_bytes());
        bytes[42..44].copy_from_slice(&32u16.to_le_bytes());
        bytes[44..46].copy_from_slice(&1u16.to_le_bytes());
        bytes[52..56].copy_from_slice(&1u32.to_le_bytes());
        bytes[56..60].copy_from_slice(&0x100u32.to_le_bytes());
        bytes[60..64].copy_from_slice(&0x0880_0000u32.to_le_bytes());
        bytes[68..72].copy_from_slice(&8u32.to_le_bytes());
        bytes[72..76].copy_from_slice(&16u32.to_le_bytes());
        bytes[76..80].copy_from_slice(&7u32.to_le_bytes());
        bytes[0x100..].copy_from_slice(&[5, 0, 1, 0, 0, 0, 0, 0]);
        bytes
    }
    #[test]
    fn rejects_junk() {
        assert!(inspect(b"nope").is_err())
    }
    #[test]
    fn rejects_wrong_elf_class() {
        let mut b = vec![0; 64];
        b[..4].copy_from_slice(b"\x7fELF");
        b[4] = 2;
        b[5] = 1;
        assert!(inspect(&b).is_err())
    }
    #[test]
    fn maps_synthetic_elf_and_zero_fills_bss() {
        let bytes = synthetic_elf();
        let mut memory = Memory::default();
        assert_eq!(map_elf(&bytes, &mut memory).unwrap(), 0x0880_0000);
        assert_eq!(memory.read_u32(0x0880_0000).unwrap(), 0x0001_0005);
        assert_eq!(memory.read_u32(0x0880_0008).unwrap(), 0);
        assert_eq!(memory.fetch_u32(0x0880_0000).unwrap(), 0x0001_0005);
    }

    struct CountingReader {
        cursor: Cursor<Vec<u8>>,
        reads: Rc<Cell<usize>>,
    }

    impl Read for CountingReader {
        fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
            self.reads.set(self.reads.get() + 1);
            self.cursor.read(output)
        }
    }

    impl Seek for CountingReader {
        fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
            self.cursor.seek(position)
        }
    }

    #[test]
    fn raw_reads_reuse_sequential_read_ahead() {
        let reads = Rc::new(Cell::new(0));
        let data = (0..0x4000).map(|value| value as u8).collect::<Vec<_>>();
        let mut image = IsoImage {
            file: Box::new(CountingReader {
                cursor: Cursor::new(data.clone()),
                reads: Rc::clone(&reads),
            }),
            image_len: data.len() as u64,
            root: IsoEntry {
                name: String::new(),
                extent: 0,
                size: 0,
                directory: true,
            },
            raw_cache: None,
        };

        assert_eq!(image.read_raw(0x100, 32).unwrap(), data[0x100..0x120]);
        let reads_after_first = reads.get();
        assert_eq!(image.read_raw(0x120, 32).unwrap(), data[0x120..0x140]);
        assert_eq!(reads.get(), reads_after_first);
    }
}
