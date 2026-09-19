use crate::Result;
use crate::ffi::{check, last_error};
use shadertoy_sys as sys;
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::rc::Rc;

/// Display-less OpenGL context suitable for deterministic/offscreen ShaderToy rendering.
///
/// On Linux the native helper creates a surfaceless EGL context directly, so no X11/Wayland
/// display is required. Context creation and destruction must happen on the process main
/// thread. The context itself is thread-affine, so this type is deliberately neither Send
/// nor Sync and must be created and dropped on that main thread.
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
