# LowRankArena project page

This directory contains the static project page for LowRankArena. It has no
build-time dependencies: `index.html`, `styles.css`, `data.js`, and `app.js`
can be served directly.

To preview the page locally from the repository root:

```bash
python -m http.server 8000
```

Then open <http://localhost:8000/docs/>.

For GitHub Pages, configure the repository to deploy from the `main` branch and
the `/docs` folder. The expected public URL is
<https://zishan-shao.github.io/lowrankarena/>.

The public page exposes paper, code, checkpoint, result, reproduction, and
citation links. Citation metadata identifies the arXiv paper as
[`2608.26389`](https://arxiv.org/abs/2608.26389).
