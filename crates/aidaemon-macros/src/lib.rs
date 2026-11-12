//! The `#[tool]` attribute macro for aidaemon agents.
//!
//! Use it through `aidaemon-sdk` (`aidaemon_sdk::tool`), which carries the
//! runtime support the generated code needs (serde, schemars and the
//! argument checks). The full reference is on `aidaemon_sdk::tool`.

use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::{format_ident, quote};
use syn::parse::Parser;
use syn::spanned::Spanned;
use syn::{Attribute, Expr, ExprLit, FnArg, Ident, ItemFn, Lit, LitStr, Meta, Pat, Path, Type};

/// Turn a function into a type implementing `aidaemon_core::Tool`.
///
/// See `aidaemon_sdk::tool` for the full documentation and examples.
#[proc_macro_attribute]
pub fn tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    let function =
        match syn::parse::<ItemFn>(item) {
            Ok(function) => function,
            Err(_) => return syn::Error::new(
                Span::call_site(),
                "#[tool] goes on a function: `#[tool(read_only)] async fn name(arg: Type) -> ...`",
            )
            .to_compile_error()
            .into(),
        };
    match expand(attr.into(), function) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

/// The macro's arguments.
#[derive(Default)]
struct Options {
    preset: Option<(Preset, Span)>,
    idempotent: Option<bool>,
    egress: Option<bool>,
    trusted: bool,
    name: Option<LitStr>,
    description: Option<LitStr>,
    timeout_secs: Option<u64>,
    access: Option<Path>,
    effects: Option<Path>,
    krate: Option<Path>,
}

#[derive(Clone, Copy, PartialEq)]
enum Preset {
    ReadOnly,
    Write,
    Destructive,
}

impl Options {
    fn parse(attr: TokenStream2) -> syn::Result<Self> {
        let mut options = Options::default();
        let parser = syn::meta::parser(|meta| {
            let key = meta
                .path
                .get_ident()
                .map(Ident::to_string)
                .unwrap_or_default();
            let mut set_preset = |preset: Preset| {
                if let Some((_, span)) = options.preset {
                    let mut err = syn::Error::new(
                        meta.path.span(),
                        "pick one of `read_only`, `write` and `destructive`",
                    );
                    err.combine(syn::Error::new(span, "already set here"));
                    return Err(err);
                }
                options.preset = Some((preset, meta.path.span()));
                Ok(())
            };
            match key.as_str() {
                "read_only" => set_preset(Preset::ReadOnly),
                "write" => set_preset(Preset::Write),
                "destructive" => set_preset(Preset::Destructive),
                "idempotent" => {
                    options.idempotent = Some(flag(&meta)?);
                    Ok(())
                }
                "egress" | "open_world" => {
                    options.egress = Some(flag(&meta)?);
                    Ok(())
                }
                "trusted" => {
                    options.trusted = flag(&meta)?;
                    Ok(())
                }
                "name" => {
                    options.name = Some(meta.value()?.parse()?);
                    Ok(())
                }
                "description" => {
                    options.description = Some(meta.value()?.parse()?);
                    Ok(())
                }
                "timeout_secs" => {
                    let lit: syn::LitInt = meta.value()?.parse()?;
                    options.timeout_secs = Some(lit.base10_parse()?);
                    Ok(())
                }
                "access" => {
                    options.access = Some(meta.value()?.parse()?);
                    Ok(())
                }
                "effects" => {
                    options.effects = Some(meta.value()?.parse()?);
                    Ok(())
                }
                "crate" => {
                    options.krate = Some(meta.value()?.parse()?);
                    Ok(())
                }
                _ => Err(meta.error(
                    "unknown #[tool] argument; expected one of: read_only, write, destructive, \
                     idempotent, egress, open_world, trusted, name, description, timeout_secs, \
                     access, effects, crate",
                )),
            }
        });
        parser.parse2(attr)?;
        Ok(options)
    }
}

/// `key` alone is true; `key = true|false` is explicit.
fn flag(meta: &syn::meta::ParseNestedMeta) -> syn::Result<bool> {
    if meta.input.peek(syn::Token![=]) {
        let lit: syn::LitBool = meta.value()?.parse()?;
        Ok(lit.value)
    } else {
        Ok(true)
    }
}

enum Param {
    /// A model-facing argument, a field of the generated args struct.
    Field {
        ident: Ident,
        ty: Box<Type>,
        attrs: Vec<Attribute>,
    },
    /// `#[args] name: Type`: the whole argument object.
    Args { ty: Box<Type> },
    /// `ctx: &ToolContext`.
    Context,
}

fn expand(attr: TokenStream2, mut function: ItemFn) -> syn::Result<TokenStream2> {
    let options = Options::parse(attr)?;
    let krate = options
        .krate
        .clone()
        .unwrap_or_else(|| syn::parse_quote!(::aidaemon_sdk));
    let private = quote!(#krate::__private);
    let core = quote!(#private::aidaemon_core);

    if !function.sig.generics.params.is_empty() {
        return Err(syn::Error::new(
            function.sig.generics.span(),
            "#[tool] functions cannot be generic",
        ));
    }

    let fn_ident = function.sig.ident.clone();
    let description = match &options.description {
        Some(lit) => lit.value(),
        None => doc_text(&function.attrs),
    };
    if description.trim().is_empty() {
        return Err(syn::Error::new(
            fn_ident.span(),
            "a tool needs a description: write a doc comment on the function \
             (the model chooses tools by it) or pass `description = \"...\"`",
        ));
    }
    let tool_name = options
        .name
        .as_ref()
        .map(LitStr::value)
        .unwrap_or_else(|| fn_ident.to_string().trim_start_matches("r#").to_string());
    let tool_name_valid = (1..=64).contains(&tool_name.len())
        && tool_name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-');
    if !tool_name_valid {
        return Err(syn::Error::new(
            options
                .name
                .as_ref()
                .map(|n| n.span())
                .unwrap_or(fn_ident.span()),
            "tool names are 1-64 ASCII letters, digits, '_' or '-'",
        ));
    }

    let mut params = Vec::new();
    for input in function.sig.inputs.iter_mut() {
        let FnArg::Typed(pat_type) = input else {
            return Err(syn::Error::new(
                input.span(),
                "#[tool] goes on a free function; pass state through \
                 `ToolContext::extensions` instead of `self`",
            ));
        };
        let is_args = pat_type.attrs.iter().any(|a| a.path().is_ident("args"));
        let attrs = std::mem::take(&mut pat_type.attrs);
        if is_context(&pat_type.ty) {
            params.push(Param::Context);
            continue;
        }
        if is_args {
            params.push(Param::Args {
                ty: pat_type.ty.clone(),
            });
            continue;
        }
        let Pat::Ident(pat_ident) = &*pat_type.pat else {
            return Err(syn::Error::new(
                pat_type.pat.span(),
                "tool arguments need a plain name (it becomes the JSON key)",
            ));
        };
        let field_attrs = attrs
            .into_iter()
            .filter(|a| {
                a.path().is_ident("doc")
                    || a.path().is_ident("serde")
                    || a.path().is_ident("schemars")
            })
            .collect();
        params.push(Param::Field {
            ident: pat_ident.ident.clone(),
            ty: pat_type.ty.clone(),
            attrs: field_attrs,
        });
    }
    let args_params = params
        .iter()
        .filter(|p| matches!(p, Param::Args { .. }))
        .count();
    let field_params = params
        .iter()
        .filter(|p| matches!(p, Param::Field { .. }))
        .count();
    if args_params > 1 || (args_params == 1 && field_params > 0) {
        return Err(syn::Error::new(
            function.sig.inputs.span(),
            "an `#[args]` parameter takes the whole argument object: \
             it must be the only argument besides `&ToolContext`",
        ));
    }

    let tool_ident = format_ident!(
        "{}",
        pascal_case(&fn_ident.to_string()),
        span = fn_ident.span()
    );
    let args_struct = format_ident!("__{}Args", tool_ident);
    let serde_path = LitStr::new(&quote!(#private::serde).to_string(), Span::call_site());
    let schemars_path = LitStr::new(&quote!(#private::schemars).to_string(), Span::call_site());

    let (args_type, args_decl, call_args) = if let Some(ty) = params.iter().find_map(|p| match p {
        Param::Args { ty } => Some(ty.clone()),
        _ => None,
    }) {
        let call_args: Vec<TokenStream2> = params
            .iter()
            .map(|p| match p {
                Param::Context => quote!(ctx),
                _ => quote!(args),
            })
            .collect();
        (quote!(#ty), quote!(), call_args)
    } else {
        let fields = params.iter().filter_map(|p| match p {
            Param::Field { ident, ty, attrs } => Some(quote!(#(#attrs)* #ident: #ty,)),
            _ => None,
        });
        let decl = quote! {
            #[doc(hidden)]
            #[allow(non_camel_case_types, dead_code)]
            #[derive(#private::serde::Deserialize, #private::schemars::JsonSchema)]
            #[serde(crate = #serde_path, deny_unknown_fields)]
            #[schemars(crate = #schemars_path)]
            struct #args_struct {
                #(#fields)*
            }
        };
        let call_args = params
            .iter()
            .map(|p| match p {
                Param::Context => quote!(ctx),
                Param::Field { ident, .. } => quote!(args.#ident),
                Param::Args { .. } => unreachable!(),
            })
            .collect();
        (quote!(#args_struct), decl, call_args)
    };

    // Annotations: no preset is the most conservative value, the same as a
    // hand-written tool's `ToolAnnotations::default()`.
    let preset = options
        .preset
        .map(|(p, _)| p)
        .unwrap_or(Preset::Destructive);
    let base = match preset {
        Preset::ReadOnly => quote!(#core::ToolAnnotations::READ_ONLY),
        Preset::Write => quote!(#core::ToolAnnotations::WRITE),
        Preset::Destructive => quote!(#core::ToolAnnotations::DESTRUCTIVE),
    };
    let idempotent = options
        .idempotent
        .map(|v| quote!(annotations.idempotent = #v;));
    let egress = options.egress.map(|v| quote!(annotations.egress = #v;));
    let trust = if options.trusted {
        quote!(#core::SourceTrust::Trusted)
    } else {
        quote!(#core::SourceTrust::Untrusted)
    };
    let timeout = options.timeout_secs.map(|secs| {
        quote! {
            fn timeout(&self) -> ::core::option::Option<::std::time::Duration> {
                ::core::option::Option::Some(::std::time::Duration::from_secs(#secs))
            }
        }
    });
    let access = options.access.as_ref().map(|path| {
        quote! {
            fn access(&self, arguments: &#private::serde_json::Value) -> #core::AccessManifest {
                match #private::parse_arguments::<#args_type>(arguments) {
                    ::core::result::Result::Ok(args) => #path(&args),
                    ::core::result::Result::Err(_) => #core::AccessManifest::default(),
                }
            }
        }
    });
    let effects = options.effects.as_ref().map(|path| {
        quote! {
            fn effects(&self, arguments: &#private::serde_json::Value) -> #core::CallEffects {
                match #private::parse_arguments::<#args_type>(arguments) {
                    ::core::result::Result::Ok(args) => #path(&args),
                    ::core::result::Result::Err(_) => #core::CallEffects::unknown(),
                }
            }
        }
    });
    let await_call = if function.sig.asyncness.is_some() {
        quote!(.await)
    } else {
        quote!()
    };
    let vis = &function.vis;
    let tool_doc = format!("The `{tool_name}` tool, made by `#[tool]` from [`{fn_ident}`].");

    Ok(quote! {
        #function

        #args_decl

        #[doc = #tool_doc]
        #[derive(Debug, Clone, Copy, Default)]
        #vis struct #tool_ident;

        #[#private::async_trait::async_trait]
        impl #core::Tool for #tool_ident {
            fn spec(&self) -> #core::ToolSpec {
                #core::ToolSpec {
                    name: ::std::string::String::from(#tool_name),
                    description: ::std::string::String::from(#description),
                    parameters: #private::parameters::<#args_type>(),
                }
            }

            fn annotations(&self) -> #core::ToolAnnotations {
                #[allow(unused_mut)]
                let mut annotations = #base;
                #idempotent
                #egress
                annotations
            }

            fn source_trust(&self, _arguments: &#private::serde_json::Value) -> #core::SourceTrust {
                #trust
            }

            fn validate(
                &self,
                arguments: &#private::serde_json::Value,
            ) -> ::core::result::Result<(), ::std::string::String> {
                #private::parse_arguments::<#args_type>(arguments).map(|_| ())
            }

            #timeout
            #access
            #effects

            async fn call(
                &self,
                arguments: #private::serde_json::Value,
                ctx: &#core::ToolContext,
            ) -> #core::ToolResult {
                let _ = &ctx;
                let args: #args_type = match #private::parse_arguments(&arguments) {
                    ::core::result::Result::Ok(args) => args,
                    ::core::result::Result::Err(err) => {
                        return ::core::result::Result::Ok(#core::ToolOutput::error(err))
                    }
                };
                #private::IntoToolResult::into_tool_result(
                    #fn_ident(#(#call_args),*) #await_call
                )
            }
        }
    })
}

/// `&ToolContext` (any path ending in `ToolContext`).
fn is_context(ty: &Type) -> bool {
    let Type::Reference(reference) = ty else {
        return false;
    };
    let Type::Path(path) = &*reference.elem else {
        return false;
    };
    path.path
        .segments
        .last()
        .is_some_and(|segment| segment.ident == "ToolContext")
}

/// The text of `///` comments, one line each, common leading space removed.
fn doc_text(attrs: &[Attribute]) -> String {
    let lines: Vec<String> = attrs
        .iter()
        .filter(|attr| attr.path().is_ident("doc"))
        .filter_map(|attr| match &attr.meta {
            Meta::NameValue(nv) => match &nv.value {
                Expr::Lit(ExprLit {
                    lit: Lit::Str(s), ..
                }) => Some(s.value()),
                _ => None,
            },
            _ => None,
        })
        .flat_map(|s| {
            if s.is_empty() {
                vec![String::new()]
            } else {
                s.lines().map(str::to_string).collect::<Vec<_>>()
            }
        })
        .collect();
    let indent = lines
        .iter()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.len() - l.trim_start().len())
        .min()
        .unwrap_or(0);
    let text = lines
        .iter()
        .map(|l| l.get(indent..).unwrap_or("").trim_end())
        .collect::<Vec<_>>()
        .join("\n");
    text.trim().to_string()
}

fn pascal_case(name: &str) -> String {
    name.trim_start_matches("r#")
        .split('_')
        .filter(|part| !part.is_empty())
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                Some(first) => first.to_ascii_uppercase().to_string() + chars.as_str(),
                None => String::new(),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_become_pascal_case() {
        assert_eq!(pascal_case("lookup_order"), "LookupOrder");
        assert_eq!(pascal_case("r#move"), "Move");
        assert_eq!(pascal_case("get_v2_data"), "GetV2Data");
    }

    #[test]
    fn doc_comments_lose_their_leading_space() {
        let function: ItemFn = syn::parse_quote! {
            /// Look up an order.
            ///
            ///   Indented detail.
            fn f() {}
        };
        assert_eq!(
            doc_text(&function.attrs),
            "Look up an order.\n\n  Indented detail."
        );
    }

    #[test]
    fn a_tool_without_a_description_is_refused() {
        let function: ItemFn = syn::parse_quote! { async fn f(x: String) -> String { x } };
        let err = expand(quote!(read_only), function).unwrap_err();
        assert!(err.to_string().contains("description"), "{err}");
    }

    #[test]
    fn two_presets_are_refused() {
        let err = Options::parse(quote!(read_only, destructive))
            .err()
            .unwrap();
        assert!(err.to_string().contains("pick one"), "{err}");
    }

    #[test]
    fn unknown_arguments_are_refused() {
        let err = Options::parse(quote!(readonly)).err().unwrap();
        assert!(err.to_string().contains("unknown"), "{err}");
    }
}
