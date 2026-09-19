use std::ffi::NulError;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("string contains an interior NUL byte")]
    Nul(#[from] NulError),
    #[error("{0}")]
    Native(String),
    #[error("buffer size does not match {width}x{height} RGBA8 image")]
    InvalidRgbaBuffer { width: u32, height: u32 },
}

pub type Result<T> = std::result::Result<T, Error>;
