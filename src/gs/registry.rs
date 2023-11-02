use metal::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub struct Library {
    functions: HashMap<String, (Function, ComputePipelineState)>,
}

impl Library {
    pub fn new(device: &Device, lib: metal::Library) -> Library {
        let mut functions = HashMap::<String, (Function, ComputePipelineState)>::new();
        for function_name in lib.function_names().iter() {
            let function = lib.get_function(function_name, None).unwrap();
            let pipeline_state = device
                .new_compute_pipeline_state_with_function(&function)
                .unwrap();

            functions.insert(function_name.to_string(), (function, pipeline_state));
        }

        Library { functions }
    }
}

pub struct ProgramRegistry<'a> {
    device: &'a Device,
    libraries: HashMap<String, Library>,
}

impl<'a> ProgramRegistry<'_> {
    pub fn new(device: &'a Device) -> ProgramRegistry<'a> {
        ProgramRegistry {
            device,
            libraries: HashMap::new(),
        }
    }

    pub fn register_library(
        &mut self,
        library_name: &str,
        path: &Path,
        compile_options: Option<&CompileOptions>,
    ) {
        let src = std::fs::read_to_string(path).unwrap();
        let mtl_lib_result = self
            .device
            .new_library_with_source(&src, compile_options.unwrap_or(&CompileOptions::new()));
        let mtl_lib = match mtl_lib_result {
            Ok(lib) => lib,
            Err(error) => panic!(
                "Compiling library {} failed with message:\n{}",
                path.display(),
                error
            ),
        };
        self.libraries
            .insert(library_name.to_owned(), Library::new(self.device, mtl_lib));
    }

    pub fn get_function(
        &self,
        library_name: &str,
        function_name: &str,
    ) -> Option<&(Function, ComputePipelineState)> {
        self.libraries
            .get(library_name)?
            .functions
            .get(function_name)
    }
}

static NAMES: &'static [&'static str] = &["sum", "sort"];

fn default_registry_path() -> PathBuf {
    if let Ok(path) = std::env::var("REGISTRY_PATH") {
        PathBuf::from(path)
    } else {
        PathBuf::from(".")
    }
}

pub fn create_default_registry<'a>(
    device: &'a Device,
    folder: Option<&Path>,
) -> ProgramRegistry<'a> {
    let mut registry = ProgramRegistry::new(&device);

    let resolved_folder = if let Some(path) = folder {
        PathBuf::from(path)
    } else {
        default_registry_path()
    };

    for name in NAMES {
        let file_path = resolved_folder.join(format!("{}.metal", name));
        registry.register_library(name, &file_path, None);
    }

    registry
}
