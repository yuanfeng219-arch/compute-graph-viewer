/**
 * build.js — 把 demo/index.html 及所有本地资源打包成单个自包含 HTML 文件
 * 用法：node build.js
 * 输出：demo/dist.html
 */

const fs   = require('fs');
const path = require('path');

const SRC  = path.join(__dirname, 'index.html');
const DIST = path.join(__dirname, 'dist.html');

const MIME = {
  '.svg':  'image/svg+xml',
  '.png':  'image/png',
  '.jpg':  'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif':  'image/gif',
  '.webp': 'image/webp',
};

function toDataUri(filepath) {
  const ext  = path.extname(filepath).toLowerCase();
  const mime = MIME[ext] || 'application/octet-stream';
  const data = fs.readFileSync(filepath).toString('base64');
  return `data:${mime};base64,${data}`;
}

function inlineAssets(html, baseDir) {
  const replaced = new Set();

  // <img src="file"> 和 <img src='file'>
  html = html.replace(/src=(['"])(?!data:)([^'"]+\.(svg|png|jpg|jpeg|gif|webp))\1/gi,
    (match, quote, filename) => {
      const filepath = path.join(baseDir, filename);
      if (!fs.existsSync(filepath)) return match;
      replaced.add(filename);
      return `src=${quote}${toDataUri(filepath)}${quote}`;
    }
  );

  // url('file') 和 url("file") 和 url(file)  ← CSS/JS 里的资源引用
  html = html.replace(/url\((['"]?)(?!data:)([^)'"\s]+\.(svg|png|jpg|jpeg|gif|webp))\1\)/gi,
    (match, quote, filename) => {
      const filepath = path.join(baseDir, filename);
      if (!fs.existsSync(filepath)) return match;
      replaced.add(filename);
      return `url(${quote}${toDataUri(filepath)}${quote})`;
    }
  );

  return { html, replaced };
}

const baseDir = path.dirname(SRC);
let html = fs.readFileSync(SRC, 'utf-8');

const { html: inlined, replaced } = inlineAssets(html, baseDir);
fs.writeFileSync(DIST, inlined, 'utf-8');

const sizeKB = (fs.statSync(DIST).size / 1024).toFixed(1);
console.log(`✓ 内嵌资源: ${replaced.size ? [...replaced].join(', ') : '（无）'}`);
console.log(`✓ 输出: ${DIST}  (${sizeKB} KB)`);
