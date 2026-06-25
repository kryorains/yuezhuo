mod ast;
mod codegen;
mod ir;
mod lexer;
mod parser;
mod token;

use lexer::Lexer;
use parser::Parser;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;
use token::Token;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "Usage: {} <input.sysy> [output.s] [-S|--ir] [-o <out>] [-O0|-O1|-O2] [--target x86_64|aarch64|riscv64] [--lex]",
            args[0]
        );
        process::exit(2);
    }

    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut target = default_target();
    let mut lex_only = false;
    let mut emit_asm = false; // -S
    let mut emit_ir = false;
    let mut opt_o1 = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-o" => {
                i += 1;
                if i >= args.len() {
                    panic!("-o requires a path");
                }
                output = Some(PathBuf::from(&args[i]));
            }
            "-S" => {
                emit_asm = true;
            }
            "--ir" => {
                emit_ir = true;
            }
            "-O0" => {}
            "-O1" | "-O2" => {
                opt_o1 = true;
            }
            "--target" => {
                i += 1;
                if i >= args.len() {
                    panic!("--target requires a value");
                }
                target = parse_target(&args[i]);
            }
            s if s.starts_with("--target=") => {
                let v = s.trim_start_matches("--target=");
                target = parse_target(v);
            }
            "--lex" => {
                lex_only = true;
            }
            s if s.starts_with('-') => {
                panic!("Unknown flag: {}", s);
            }
            s => {
                if input.is_none() {
                    input = Some(PathBuf::from(s));
                } else if output.is_none() && !emit_ir && !lex_only {
                    output = Some(PathBuf::from(s));
                    emit_asm = true;
                } else {
                    panic!("Unexpected extra arg: {}", s);
                }
            }
        }
        i += 1;
    }

    let input = input.unwrap_or_else(|| panic!("Missing input.sysy"));
    let source =
        fs::read_to_string(&input).unwrap_or_else(|e| panic!("Read {:?} failed: {}", input, e));

    if lex_only {
        let mut lexer = Lexer::new(&source);
        loop {
            let tok = lexer.next_token();
            println!("{:?}", tok);
            if tok == Token::Eof {
                break;
            }
        }
        return;
    }

    if !emit_asm && !emit_ir {
        panic!("Please pass -S or --ir");
    }
    let output = output.unwrap_or_else(|| panic!("Missing -o <out>"));

    let mut parser = Parser::new(&source);
    let prog = parser.parse_program();

    let mut module =
        ir::lower::lower_program(&prog).unwrap_or_else(|e| panic!("IR lower failed: {:?}", e));
    let opt_level = if opt_o1 {
        ir::pass::OptLevel::O1
    } else {
        ir::pass::OptLevel::O0
    };
    ir::pass::run_pipeline(&mut module, opt_level);

    if emit_ir {
        fs::write(&output, format!("{:#?}", module))
            .unwrap_or_else(|e| panic!("Write {:?} failed: {}", output, e));
        eprintln!("Wrote {:?}", output);
        return;
    }

    let asm = codegen::emit_asm(target, &module);
    fs::write(&output, asm).unwrap_or_else(|e| panic!("Write {:?} failed: {}", output, e));
    eprintln!("Wrote {:?}", output);
}

fn parse_target(s: &str) -> codegen::Target {
    match s {
        "x86_64" | "x86-64" | "amd64" => codegen::Target::X86_64,
        "aarch64" | "arm64" => codegen::Target::AArch64,
        "riscv64" | "riscv64gc" => codegen::Target::Riscv64,
        _ => panic!("Unknown target: {}", s),
    }
}

fn default_target() -> codegen::Target {
    match env::consts::ARCH {
        "aarch64" => codegen::Target::AArch64,
        "riscv64" => codegen::Target::Riscv64,
        "x86_64" => codegen::Target::X86_64,
        _ => codegen::Target::X86_64,
    }
}
