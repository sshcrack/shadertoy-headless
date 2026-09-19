#![allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    unsafe_op_in_unsafe_fn
)]
#![doc = "Raw, bindgen-generated bindings to ShaderToy's stable C ABI."]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
