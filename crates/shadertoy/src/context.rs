use crate::Result;
use crate::ffi::{check, last_error};
use shadertoy_sys as sys;
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::rc::Rc;

/// Hidden OpenGL context suitable for deterministic/offscreen ShaderToy rendering.
///
/// OpenGL contexts are thread-affine. This type is deliberately neither Send nor Sync.
pub struct HeadlessContext {
    handle: NonNull<sys::st_context>,
    _thread_affine: PhantomData<Rc<()>>,
}

impl HeadlessContext {
    pub fn new(width: u32, height: u32) -> Result<Self> {
        // SAFETY: generated binding calls the repository-owned stable C ABI.
        let handle = unsafe { sys::st_context_create_hidden(width, height) };
        let handle = NonNull::new(handle).ok_or_else(last_error)?;
        Ok(Self {
            handle,
            _thread_affine: PhantomData,
        })
    }

    pub fn make_current(&self) -> Result<()> {
        // SAFETY: handle is valid for self's lifetime and destroyed exactly once in Drop.
        check(unsafe { sys::st_context_make_current(self.handle.as_ptr()) })
    }
}

impl Drop for HeadlessContext {
    fn drop(&mut self) {
        // SAFETY: handle came from st_context_create_hidden and is owned by self.
        unsafe { sys::st_context_destroy(self.handle.as_ptr()) };
    }
}
