use crate::ffi::{check, filter_kind, input_kind, last_error, pass_kind, wrap_kind};
use crate::types::checked_image_len;
use crate::{Error, Filter, InputKind, PassKind, Result, Wrap};
use shadertoy_sys as sys;
use std::ffi::CString;
use std::ptr::NonNull;

/// Native project graph builder backed by the canonical C++ project semantics.
pub struct Project {
    handle: NonNull<sys::st_project>,
}

impl Project {
    pub fn new(name: &str) -> Result<Self> {
        let name = CString::new(name)?;
        // SAFETY: name is NUL-terminated for the duration of the call.
        let handle = unsafe { sys::st_project_create(name.as_ptr()) };
        let handle = NonNull::new(handle).ok_or_else(last_error)?;
        Ok(Self { handle })
    }

    pub fn add_pass(&mut self, name: &str, kind: PassKind, source: &str) -> Result<&mut Self> {
        let name = CString::new(name)?;
        let source = CString::new(source)?;
        // SAFETY: project handle is valid and both C strings live across the call.
        check(unsafe {
            sys::st_project_add_pass(
                self.handle.as_ptr(),
                name.as_ptr(),
                pass_kind(kind),
                source.as_ptr(),
            )
        })?;
        Ok(self)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_input(
        &mut self,
        pass: &str,
        channel: u32,
        kind: InputKind,
        source: &str,
        previous_frame: bool,
        filter: Filter,
        wrap: Wrap,
    ) -> Result<&mut Self> {
        let pass = CString::new(pass)?;
        let source = CString::new(source)?;
        // SAFETY: project handle is valid and both C strings live across the call.
        check(unsafe {
            sys::st_project_add_input(
                self.handle.as_ptr(),
                pass.as_ptr(),
                channel,
                input_kind(kind),
                source.as_ptr(),
                i32::from(previous_frame),
                filter_kind(filter),
                wrap_kind(wrap),
            )
        })?;
        Ok(self)
    }

    pub fn add_texture_rgba8(
        &mut self,
        name: &str,
        width: u32,
        height: u32,
        rgba: &[u8],
    ) -> Result<&mut Self> {
        if rgba.len() != checked_image_len(width, height, 4)? {
            return Err(Error::InvalidRgbaBuffer { width, height });
        }
        let name = CString::new(name)?;
        // SAFETY: project handle/data are valid for the call; native implementation copies the data.
        check(unsafe {
            sys::st_project_add_texture_rgba8(
                self.handle.as_ptr(),
                name.as_ptr(),
                width,
                height,
                rgba.as_ptr(),
                rgba.len(),
            )
        })?;
        Ok(self)
    }

    pub fn add_cubemap_rgba8(&mut self, name: &str, size: u32, rgba: &[u8]) -> Result<&mut Self> {
        let face_len = checked_image_len(size, size, 4)?;
        let expected = face_len.checked_mul(6).ok_or(Error::ImageSizeOverflow {
            width: size,
            height: size,
            channels: 24,
        })?;
        if rgba.len() != expected {
            return Err(Error::InvalidRgbaBuffer {
                width: size.saturating_mul(6),
                height: size,
            });
        }
        let name = CString::new(name)?;
        // SAFETY: project handle/data are valid for the call; native implementation copies the data.
        check(unsafe {
            sys::st_project_add_cubemap_rgba8(
                self.handle.as_ptr(),
                name.as_ptr(),
                size,
                rgba.as_ptr(),
                rgba.len(),
            )
        })?;
        Ok(self)
    }

    pub fn add_volume_u8(
        &mut self,
        name: &str,
        size: u32,
        channels: u32,
        data: &[u8],
    ) -> Result<&mut Self> {
        if channels != 1 && channels != 4 {
            return Err(Error::Native("volume channels must be 1 or 4".into()));
        }
        let plane = checked_image_len(size, size, channels as usize)?;
        let expected = plane
            .checked_mul(size as usize)
            .ok_or(Error::ImageSizeOverflow {
                width: size,
                height: size,
                channels: (channels as usize).saturating_mul(size as usize),
            })?;
        if data.len() != expected {
            return Err(Error::Native("volume payload has the wrong size".into()));
        }
        let name = CString::new(name)?;
        // SAFETY: project handle/data are valid for the call; native implementation copies the data.
        check(unsafe {
            sys::st_project_add_volume_u8(
                self.handle.as_ptr(),
                name.as_ptr(),
                size,
                channels,
                data.as_ptr(),
                data.len(),
            )
        })?;
        Ok(self)
    }

    pub(crate) fn as_ptr(&self) -> *const sys::st_project {
        self.handle.as_ptr()
    }
}

impl Drop for Project {
    fn drop(&mut self) {
        // SAFETY: handle came from st_project_create and is owned by self.
        unsafe { sys::st_project_destroy(self.handle.as_ptr()) };
    }
}
