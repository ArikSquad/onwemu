//! Small, sandboxed path resolver for the guest's `ms0:` and optional `host0:`
//! mounts.
//!
//! The resolver never opens a file. It only turns a guest path into a host path
//! after rejecting traversal components, leaving the caller to decide which
//! file operation is appropriate.

use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
/// Errors returned while resolving a guest mount path.
pub enum VfsError {
    #[error("path escapes its mount")]
    /// The path contains a parent component that would leave its mount.
    Traversal,
    #[error("unknown mount")]
    /// The path uses a mount that was not configured.
    UnknownMount,
}
#[derive(Clone, Debug)]
/// Host roots used for the PSP memory-stick and optional host mounts.
pub struct VirtualFileSystem {
    ms0: PathBuf,
    host0: Option<PathBuf>,
}
impl VirtualFileSystem {
    /// Create a resolver with an `ms0:` root and an optional `host0:` root.
    pub fn new(ms0: PathBuf, host0: Option<PathBuf>) -> Self {
        Self { ms0, host0 }
    }
    /// Resolve a PSP path without allowing it to escape its mount root.
    ///
    /// Both slash styles are accepted because PSP paths commonly use `/` while
    /// host tooling and saved metadata sometimes use `\\`.
    pub fn resolve(&self, guest: &str) -> Result<PathBuf, VfsError> {
        let (mount, rest) = guest.split_once(':').ok_or(VfsError::UnknownMount)?;
        let root = match mount.to_ascii_lowercase().as_str() {
            "ms0" => &self.ms0,
            "host0" => self.host0.as_ref().ok_or(VfsError::UnknownMount)?,
            _ => return Err(VfsError::UnknownMount),
        };
        let mut out = root.clone();
        for component in rest.trim_start_matches(['/', '\\']).split(['/', '\\']) {
            if component.is_empty() || component == "." {
                continue;
            }
            if component == ".." {
                return Err(VfsError::Traversal);
            }
            out.push(component);
        }
        Ok(out)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sandbox_rejects_traversal_and_accepts_both_guest_slash_styles() {
        let v = VirtualFileSystem::new("safe".into(), None);
        assert_eq!(
            v.resolve("ms0:/SAVE/a.bin").unwrap(),
            PathBuf::from("safe/SAVE/a.bin")
        );
        assert_eq!(
            v.resolve("MS0:\\SAVE\\a.bin").unwrap(),
            PathBuf::from("safe/SAVE/a.bin")
        );
        assert!(v.resolve("ms0:/../secret").is_err())
    }

    #[test]
    fn mount_lookup_distinguishes_optional_host0_from_unknown_mounts() {
        let without_host = VirtualFileSystem::new("safe".into(), None);
        assert!(matches!(
            without_host.resolve("host0:/file"),
            Err(VfsError::UnknownMount)
        ));
        assert!(matches!(
            without_host.resolve("umd:/file"),
            Err(VfsError::UnknownMount)
        ));
        assert!(matches!(
            without_host.resolve("ms0"),
            Err(VfsError::UnknownMount)
        ));

        let with_host = VirtualFileSystem::new("safe".into(), Some("host".into()));
        assert_eq!(
            with_host.resolve("host0:/file").unwrap(),
            PathBuf::from("host/file")
        );
    }
}
