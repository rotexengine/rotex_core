use std::{env, fs, path::PathBuf};

use shaderc::{Compiler, ShaderKind};

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let shader_dir = PathBuf::from("examples/shaders");

    compile_shader(
        &shader_dir.join("triangle.vert"),
        ShaderKind::Vertex,
        &out_dir.join("triangle.vert.spv"),
    );
    compile_shader(
        &shader_dir.join("triangle.frag"),
        ShaderKind::Fragment,
        &out_dir.join("triangle.frag.spv"),
    );
    compile_shader(
        &shader_dir.join("cube.vert"),
        ShaderKind::Vertex,
        &out_dir.join("cube.vert.spv"),
    );
    compile_shader(
        &shader_dir.join("cube.frag"),
        ShaderKind::Fragment,
        &out_dir.join("cube.frag.spv"),
    );

    println!("cargo:rerun-if-changed=examples/shaders");
    println!("cargo:rerun-if-changed=build.rs");
}

fn compile_shader(path: &std::path::Path, kind: ShaderKind, out_path: &std::path::Path) {
    let source = fs::read_to_string(path)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()));

    let compiler = Compiler::new().expect("failed to create shaderc compiler");
    let mut options = shaderc::CompileOptions::new().expect("failed to create shaderc options");
    options.set_optimization_level(shaderc::OptimizationLevel::Performance);

    let artifact = compiler
        .compile_into_spirv(
            &source,
            kind,
            path.file_name()
                .and_then(|name| name.to_str())
                .expect("shader path must be valid utf-8"),
            "main",
            Some(&options),
        )
        .unwrap_or_else(|err| panic!("failed to compile {}: {err}", path.display()));

    fs::write(out_path, artifact.as_binary_u8())
        .unwrap_or_else(|err| panic!("failed to write {}: {err}", out_path.display()));
}
