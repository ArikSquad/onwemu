use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::PathBuf;

const OP_NAME: u16 = 5;
const OP_TYPE_POINTER: u16 = 32;
const OP_TYPE_SAMPLED_IMAGE: u16 = 27;
const OP_VARIABLE: u16 = 59;
const OP_LOAD: u16 = 61;
const OP_DECORATE: u16 = 71;
const OP_SAMPLED_IMAGE: u16 = 86;
const OP_IMAGE_SAMPLE_IMPLICIT_LOD: u16 = 87;
const DECORATION_BINDING: u32 = 33;
const DECORATION_DESCRIPTOR_SET: u32 = 34;

pub fn run() {
    println!("cargo:rerun-if-changed=shaders/psp.wgsl");
    let output = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set"));
    let source = include_str!("shaders/psp.wgsl");
    let module = naga::front::wgsl::parse_str(source).unwrap_or_else(|error| {
        panic!("failed to parse PSP GPU WGSL: {error}");
    });
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|error| panic!("failed to validate PSP GPU WGSL: {error}"));
    for (entry_point, stage, name) in [
        ("vertex_main", naga::ShaderStage::Vertex, "psp_vertex.spv"),
        (
            "fragment_main",
            naga::ShaderStage::Fragment,
            "psp_fragment.spv",
        ),
    ] {
        let words = naga::back::spv::write_vec(
            &module,
            &info,
            &naga::back::spv::Options::default(),
            Some(&naga::back::spv::PipelineOptions {
                shader_stage: stage,
                entry_point: entry_point.into(),
            }),
        )
        .unwrap_or_else(|error| panic!("failed to compile {entry_point} to SPIR-V: {error}"));
        let words = if stage == naga::ShaderStage::Fragment {
            combine_fragment_texture_sampler(words)
        } else {
            words
        };
        let bytes = words
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        fs::write(output.join(name), bytes).unwrap_or_else(|error| {
            panic!("failed to write {name}: {error}");
        });
    }
}

fn opcode(instruction: &[u32]) -> u16 {
    (instruction[0] & 0xffff) as u16
}

/// SDL GPU's Vulkan backend consumes one combined sampler2D descriptor for a
/// `TextureSamplerBinding`. Naga's WGSL backend correctly emits separate image
/// and sampler variables, so fold this fixed pair into the descriptor shape
/// expected by SDL without requiring a C/C++ shader compiler at build time.
fn combine_fragment_texture_sampler(words: Vec<u32>) -> Vec<u32> {
    assert!(words.len() >= 5, "Naga emitted an incomplete SPIR-V module");
    let mut instructions = Vec::new();
    let mut offset = 5;
    while offset < words.len() {
        let word_count = (words[offset] >> 16) as usize;
        assert!(
            word_count >= 1 && offset + word_count <= words.len(),
            "Naga emitted a malformed SPIR-V instruction"
        );
        instructions.push(words[offset..offset + word_count].to_vec());
        offset += word_count;
    }

    let mut decorations = HashMap::<u32, (Option<u32>, Option<u32>)>::new();
    for instruction in &instructions {
        if opcode(instruction) != OP_DECORATE || instruction.len() < 4 {
            continue;
        }
        let entry = decorations.entry(instruction[1]).or_default();
        match instruction[2] {
            DECORATION_DESCRIPTOR_SET => entry.0 = Some(instruction[3]),
            DECORATION_BINDING => entry.1 = Some(instruction[3]),
            _ => {}
        }
    }

    let image_variable = decorations
        .iter()
        .find_map(|(&id, &(set, binding))| (set == Some(2) && binding == Some(0)).then_some(id))
        .expect("fragment shader is missing SDL texture binding set 2/binding 0");
    let sampler_variable = decorations
        .iter()
        .find_map(|(&id, &(set, binding))| (set == Some(2) && binding == Some(1)).then_some(id))
        .expect("fragment shader is missing WGSL sampler binding set 2/binding 1");

    let mut image_pointer = 0;
    let mut sampler_pointer = 0;
    let mut image_type = 0;
    let mut sampler_type = 0;
    let mut image_variable_instruction = None;
    for instruction in &instructions {
        if opcode(instruction) != OP_VARIABLE {
            continue;
        }
        if instruction[2] == image_variable {
            image_pointer = instruction[1];
            image_variable_instruction = Some(instruction.clone());
        } else if instruction[2] == sampler_variable {
            sampler_pointer = instruction[1];
        }
    }
    for instruction in &instructions {
        if opcode(instruction) != OP_TYPE_POINTER {
            continue;
        }
        if instruction[1] == image_pointer {
            image_type = instruction[3];
        } else if instruction[1] == sampler_pointer {
            sampler_type = instruction[3];
        }
    }
    assert!(image_pointer != 0 && sampler_pointer != 0);
    assert!(image_type != 0 && sampler_type != 0);

    let sampled_image_type = instructions
        .iter()
        .find_map(|instruction| {
            (opcode(instruction) == OP_TYPE_SAMPLED_IMAGE
                && instruction.get(2) == Some(&image_type))
            .then_some(instruction[1])
        })
        .expect("fragment shader is missing a sampled-image type");
    let (image_load, _sampler_load, sampled_image_value) = instructions
        .iter()
        .find_map(|instruction| {
            (instruction.len() >= 5
                && opcode(instruction) == OP_SAMPLED_IMAGE
                && instruction[1] == sampled_image_type)
                .then(|| (instruction[3], instruction[4], instruction[2]))
        })
        .expect("fragment shader is missing its image/sampler combination");
    let image_variable_instruction = image_variable_instruction
        .expect("fragment shader is missing its texture variable declaration");

    let image_pointer_instruction = instructions
        .iter()
        .find(|instruction| {
            opcode(instruction) == OP_TYPE_POINTER && instruction[1] == image_pointer
        })
        .cloned()
        .expect("fragment shader is missing its texture pointer type");

    let mut transformed = Vec::with_capacity(instructions.len());
    let mut sample_uses = 0;
    for mut instruction in instructions {
        let instruction_opcode = opcode(&instruction);
        if instruction_opcode == OP_TYPE_POINTER && instruction[1] == image_pointer {
            continue;
        }
        if instruction_opcode == OP_VARIABLE && instruction[2] == image_variable {
            continue;
        }
        if instruction_opcode == OP_LOAD && instruction[3] == image_variable {
            instruction[1] = sampled_image_type;
        }
        if instruction_opcode == OP_IMAGE_SAMPLE_IMPLICIT_LOD
            && instruction[3] == sampled_image_value
        {
            instruction[3] = image_load;
            sample_uses += 1;
        }

        if instruction_opcode == OP_TYPE_SAMPLED_IMAGE && instruction[1] == sampled_image_type {
            transformed.push(instruction);
            let mut pointer = image_pointer_instruction.clone();
            pointer[3] = sampled_image_type;
            transformed.push(pointer);
            transformed.push(image_variable_instruction.clone());
        } else {
            transformed.push(instruction);
        }
    }
    assert_eq!(
        sample_uses, 1,
        "fragment shader sampled-image use changed shape"
    );

    transformed.retain(|instruction| {
        let instruction_opcode = opcode(instruction);
        if instruction_opcode == OP_NAME && instruction[1] == sampler_variable {
            return false;
        }
        if instruction_opcode == OP_VARIABLE && instruction[2] == sampler_variable {
            return false;
        }
        if instruction_opcode == OP_DECORATE && instruction[1] == sampler_variable {
            return false;
        }
        if instruction_opcode == OP_LOAD && instruction[3] == sampler_variable {
            return false;
        }
        if instruction_opcode == OP_SAMPLED_IMAGE && instruction[2] == sampled_image_value {
            return false;
        }
        true
    });

    let mut normalized = words[..5].to_vec();
    for instruction in transformed {
        normalized.extend(instruction);
    }
    normalized
}
