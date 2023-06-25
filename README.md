# wordshk-tools

A combination of tools for words.hk (粵典).

## Parser
```rust
/// Parse the whole words.hk CSV database into a [Dict]
pub fn parse_dict() -> Result<Dict, Box<dyn Error>>
```
Located at `/src/lib.rs`

Parses all entries marked with OK and store the results as a list of entries. This parser is the very core of this library because its output is used by other functions like `to_apple_dict`.
To boost efficiency, no regular expressions and backtracking are used. It is powered by a library called [lip](https://github.com/AlienKevin/lip) (written by myself) that provides
flexible parser combinators and supports friendly error messages.

## Example Usages
1. Parse words.hk dictionary and extract useful information
    * See `examples/parse_dict` for more details
2. Export to Apple Dictionary
    * See `examples/export_apple_dict` for more details
3. Search words.hk
    * See `examples/benchmark_search` for more details

## Source

The full up-to-date CSV database of words.hk dictionary can be downloaded from words.hk. You can request access to the CSV using this link: https://words.hk/faiman/request_data/

## License
MIT
