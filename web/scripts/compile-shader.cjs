const fs = require("node:fs");
const glslangModule = require("@webgpu/glslang");

const [, , sourcePath, outputPath] = process.argv;
if (!sourcePath || !outputPath) {
  console.error("usage: compile-shader.cjs <source.comp> <output.spv>");
  process.exit(2);
}

try {
  const source = fs.readFileSync(sourcePath, "utf8");
  const glslang = glslangModule();
  const words = glslang.compileGLSL(source, "compute", false, "1.0");
  const bytes = Buffer.from(words.buffer, words.byteOffset, words.byteLength);
  fs.writeFileSync(outputPath, bytes);
  console.log(`compiled ${bytes.byteLength} bytes with @webgpu/glslang`);
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
