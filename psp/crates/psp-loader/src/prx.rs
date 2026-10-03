//! Decryption for the original retail PSP PRX/EBOOT format.
//!
//! The PSP's KIRK engine uses AES-128-CBC with a zero IV. Early retail EBOOTs
//! contain a pre-expanded per-tag key, so they can be handled without emulating
//! the rest of KIRK. The layout and operations are documented by PSDevWiki and
//! match the independently published pspdecrypt algorithm.

use aes::Aes128;
use cbc::cipher::{BlockDecryptMut, KeyIvInit, block_padding::NoPadding};
use flate2::read::GzDecoder;
use sha1::{Digest, Sha1};
use std::io::Read;
use thiserror::Error;

type Aes128CbcDec = cbc::Decryptor<Aes128>;

const PSP_HEADER_SIZE: usize = 0x150;
const KIRK_HEADER_SIZE: usize = 0x90;
const KIRK_OFFSET: usize = 0x40;
const KIRK1_KEY: [u8; 16] = [
    0x98, 0xc9, 0x40, 0x97, 0x5c, 0x1d, 0x10, 0xe8, 0x7f, 0xe6, 0x0e, 0xa3, 0xfd, 0x03, 0xa8, 0xba,
];
const KIRK_5D_KEY: [u8; 16] = [
    0x11, 0x5a, 0x5d, 0x20, 0xd5, 0x3a, 0x8d, 0xd3, 0x9c, 0xc5, 0xaf, 0x41, 0x0f, 0x0f, 0x18, 0x6f,
];

// pre-expanded key for retail 2.xx eboots (tag c0cb167c), serialized as the
// little-endian words used by the psp.
const EBOOT_2XX_KEY_WORDS: [u32; 36] = [
    0xda8e36fa, 0x5dd97447, 0x76c19874, 0x97e57eaf, 0x1cab09bd, 0x9835bac6, 0x03d39281, 0x03b205cf,
    0x2882e734, 0xe714f663, 0xb96e2775, 0xbd8aafc7, 0x1dd3ec29, 0xeca4a16c, 0x5f69ec87, 0x85981e92,
    0x7cfcae21, 0xbae9dd16, 0xe6a97804, 0x2eee02fc, 0x61df8a3d, 0xdd310564, 0x9697e149, 0xc2453f3b,
    0xf91d8456, 0x39da6bc8, 0xb3e5fef5, 0x89c593a3, 0xfb5c8abc, 0x6c0b7212, 0xe10dd3cb, 0x98d0b2a8,
    0x5fd61847, 0xf0dc2357, 0x7701166a, 0x0f5c3b68,
];

#[derive(Debug, Error)]
pub(crate) enum PrxError {
    #[error("truncated PRX header")]
    Truncated,
    #[error("unsupported PRX tag 0x{0:08x}")]
    UnsupportedTag(u32),
    #[error("PRX header authentication failed")]
    Authentication,
    #[error("invalid KIRK header: {0}")]
    Invalid(&'static str),
    #[error("gzip decompression failed: {0}")]
    Gzip(#[from] std::io::Error),
}

fn u32le(bytes: &[u8], offset: usize) -> Result<u32, PrxError> {
    Ok(u32::from_le_bytes(
        bytes
            .get(offset..offset + 4)
            .ok_or(PrxError::Truncated)?
            .try_into()
            .unwrap(),
    ))
}

fn decrypt_cbc(key: &[u8; 16], bytes: &mut [u8]) -> Result<(), PrxError> {
    if bytes.is_empty() || bytes.len() & 15 != 0 {
        return Err(PrxError::Invalid("AES input is not block aligned"));
    }
    Aes128CbcDec::new(key.into(), (&[0u8; 16]).into())
        .decrypt_padded_mut::<NoPadding>(bytes)
        .map_err(|_| PrxError::Invalid("AES decryption failed"))?;
    Ok(())
}

pub(crate) fn decrypt(input: &[u8]) -> Result<Vec<u8>, PrxError> {
    if input.len() < PSP_HEADER_SIZE {
        return Err(PrxError::Truncated);
    }
    let tag = u32le(input, 0xd0)?;
    if tag != 0xc0cb167c {
        return Err(PrxError::UnsupportedTag(tag));
    }
    let decrypted_size =
        usize::try_from(u32le(input, 0xb0)?).map_err(|_| PrxError::Invalid("size"))?;
    let mut xor_key = [0u8; 0x90];
    for (chunk, word) in xor_key
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(EBOOT_2XX_KEY_WORDS)
    {
        chunk.copy_from_slice(&word.to_le_bytes());
    }

    let mut type_data = [0u8; PSP_HEADER_SIZE];
    type_data[..4].copy_from_slice(&input[0xd0..0xd4]);
    type_data[4..0x18].copy_from_slice(&input[0xd4..0xe8]);
    type_data[0x18..0x40].copy_from_slice(&input[0xe8..0x110]);
    type_data[0x40..0x80].copy_from_slice(&input[0x110..0x150]);
    type_data[0x80..0xd0].copy_from_slice(&input[0x80..0xd0]);
    type_data[0xd0..].copy_from_slice(&input[..0x80]);

    // the same tag was shipped in two closely related header layouts.  type 1
    // applies one additional kirk-7 transform before authenticating it.
    let authentic = |data: &[u8; PSP_HEADER_SIZE]| {
        let mut digest = Sha1::new();
        digest.update(&xor_key[..0x14]);
        digest.update(&data[0x18..]);
        digest.finalize().as_slice() == &data[4..0x18]
    };
    if !authentic(&type_data) {
        decrypt_cbc(&KIRK_5D_KEY, &mut type_data[0x10..0xb0])?;
        if !authentic(&type_data) {
            return Err(PrxError::Authentication);
        }
    }
    let kirk_block: [u8; KIRK_HEADER_SIZE] = type_data[0x40..0xd0].try_into().unwrap();

    let mut header = kirk_block;
    for i in 0..0x70 {
        header[i] = kirk_block[i] ^ xor_key[i + 0x14];
    }
    decrypt_cbc(&KIRK_5D_KEY, &mut header[..0x70])?;
    for i in 0..0x70 {
        header[i] ^= xor_key[i + 0x20];
    }

    if u32le(&header, 0x60)? != 1 {
        return Err(PrxError::Invalid("KIRK mode is not CMD1"));
    }
    let data_size =
        usize::try_from(u32le(&header, 0x70)?).map_err(|_| PrxError::Invalid("data size"))?;
    let data_offset =
        usize::try_from(u32le(&header, 0x74)?).map_err(|_| PrxError::Invalid("data offset"))?;
    let body_start = KIRK_OFFSET
        .checked_add(KIRK_HEADER_SIZE)
        .and_then(|x| x.checked_add(data_offset))
        .ok_or(PrxError::Invalid("body offset overflow"))?;
    let encrypted_size = data_size
        .checked_add(15)
        .map(|x| x & !15)
        .ok_or(PrxError::Invalid("body size overflow"))?;
    let body_end = body_start
        .checked_add(encrypted_size)
        .ok_or(PrxError::Invalid("body end overflow"))?;
    let encrypted = input.get(body_start..body_end).ok_or(PrxError::Truncated)?;

    let mut aes_key = header[..16].to_vec();
    decrypt_cbc(&KIRK1_KEY, &mut aes_key)?;
    let key: [u8; 16] = aes_key[..16].try_into().unwrap();
    let mut output = encrypted.to_vec();
    decrypt_cbc(&key, &mut output)?;
    output.truncate(decrypted_size.min(data_size));

    if output.starts_with(&[0x1f, 0x8b]) {
        let mut decoded = Vec::new();
        GzDecoder::new(output.as_slice()).read_to_end(&mut decoded)?;
        return Ok(decoded);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unknown_tag() {
        let mut bytes = vec![0; PSP_HEADER_SIZE];
        bytes[..4].copy_from_slice(b"~PSP");
        bytes[0xd0..0xd4].copy_from_slice(&0x12345678u32.to_le_bytes());
        assert!(matches!(
            decrypt(&bytes),
            Err(PrxError::UnsupportedTag(0x12345678))
        ));
    }
}
