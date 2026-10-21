use pyo3::prelude::*;

mod tokenlizer;
mod train_bpe;

/// Python extension entry point. Register your Rust functions here.
#[pymodule]
fn _rust(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<tokenlizer::Tokenlizer>()?;
    module.add_function(wrap_pyfunction!(train_bpe::train_bpe, module)?)?;
    Ok(())
}

