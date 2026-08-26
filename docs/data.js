window.LOWRANKARENA_DATA = {
  accuracyColumns: [
    { key: "keep", label: "Keep", type: "text", always: true },
    { key: "method", label: "Method", type: "text", always: true },
    { key: "wt2", label: "WT2 PPL ↓", group: "Perplexity", direction: "min", decimals: 2, summary: true },
    { key: "c4", label: "C4 PPL ↓", group: "Perplexity", direction: "min", decimals: 2, summary: true },
    { key: "boolq", label: "BoolQ", group: "General MCQ", direction: "max", decimals: 3 },
    { key: "arce", label: "ARC-E", group: "General MCQ", direction: "max", decimals: 3 },
    { key: "arcc", label: "ARC-C", group: "General MCQ", direction: "max", decimals: 3 },
    { key: "winog", label: "WinoG.", group: "General MCQ", direction: "max", decimals: 3 },
    { key: "piqa", label: "PIQA", group: "General MCQ", direction: "max", decimals: 3 },
    { key: "hellas", label: "HellaS.", group: "General MCQ", direction: "max", decimals: 3 },
    { key: "obqa", label: "OBQA", group: "General MCQ", direction: "max", decimals: 3 },
    { key: "mcq", label: "MCQ Avg. ↑", group: "Aggregate", direction: "max", decimals: 3, summary: true },
    { key: "mathqa", label: "MathQA", group: "Math", direction: "max", decimals: 3, summary: true },
    { key: "mmlum", label: "MMLU-M", group: "Math", direction: "max", decimals: 3, summary: true },
    { key: "qret", label: "Q. Ret. ↑", group: "Aggregate", direction: "max", decimals: 3, summary: true }
  ],
  accuracy: {
    "llama-1-7b": {
      label: "Llama-1-7B",
      rows: [
        { keep: "100%", method: "Dense FP", wt2: 5.67, c4: 7.20, boolq: 0.737, arce: 0.726, arcc: 0.445, winog: 0.693, piqa: 0.786, hellas: 0.749, obqa: 0.408, mcq: 0.649, mathqa: 0.261, mmlum: 0.278, qret: 1.000 },
        { keep: "80%", method: "ASVD", wt2: 8.60, c4: 11.04, boolq: 0.743, arce: 0.628, arcc: 0.388, winog: 0.669, piqa: 0.750, hellas: 0.696, obqa: 0.392, mcq: 0.609, mathqa: 0.241, mmlum: 0.211, qret: 0.787 },
        { keep: "80%", method: "SVD-LLM v1", wt2: 7.88, c4: 16.42, boolq: 0.633, arce: 0.481, arcc: 0.317, winog: 0.584, piqa: 0.681, hellas: 0.477, obqa: 0.332, mcq: 0.501, mathqa: 0.224, mmlum: 0.291, qret: 0.767 },
        { keep: "80%", method: "DoBi-SVD", wt2: 9.23, c4: 19.01, boolq: 0.620, arce: 0.500, arcc: 0.305, winog: 0.635, piqa: 0.665, hellas: 0.570, obqa: 0.385, mcq: 0.526, mathqa: 0.229, mmlum: 0.272, qret: 0.732 },
        { keep: "80%", method: "Basis Sharing", wt2: 7.74, c4: 15.51, boolq: 0.645, arce: 0.614, arcc: 0.367, winog: 0.645, piqa: 0.711, hellas: 0.633, obqa: 0.400, mcq: 0.574, mathqa: 0.239, mmlum: 0.205, qret: 0.747 },
        { keep: "80%", method: "MoDeGPT", wt2: 6.92, c4: 10.87, boolq: 0.647, arce: 0.587, arcc: 0.353, winog: 0.637, piqa: 0.724, hellas: 0.582, obqa: 0.336, mcq: 0.552, mathqa: 0.254, mmlum: 0.304, qret: 0.880 },
        { keep: "60%", method: "ASVD", wt2: 3839.83, c4: 4268.56, boolq: 0.450, arce: 0.274, arcc: 0.244, winog: 0.503, piqa: 0.528, hellas: 0.264, obqa: 0.238, mcq: 0.357, mathqa: 0.198, mmlum: 0.272, qret: 0.458 },
        { keep: "60%", method: "SVD-LLM v1", wt2: 13.74, c4: 55.03, boolq: 0.380, arce: 0.392, arcc: 0.258, winog: 0.538, piqa: 0.572, hellas: 0.350, obqa: 0.304, mcq: 0.399, mathqa: 0.222, mmlum: 0.296, qret: 0.615 },
        { keep: "60%", method: "DoBi-SVD", wt2: 15.24, c4: 48.37, boolq: 0.433, arce: 0.389, arcc: 0.268, winog: 0.594, piqa: 0.580, hellas: 0.403, obqa: 0.312, mcq: 0.426, mathqa: 0.213, mmlum: 0.225, qret: 0.560 },
        { keep: "60%", method: "Basis Sharing", wt2: 12.42, c4: 41.13, boolq: 0.503, arce: 0.472, arcc: 0.287, winog: 0.586, piqa: 0.608, hellas: 0.452, obqa: 0.344, mcq: 0.465, mathqa: 0.209, mmlum: 0.268, qret: 0.622 },
        { keep: "60%", method: "MoDeGPT", wt2: 12.43, c4: 23.40, boolq: 0.613, arce: 0.474, arcc: 0.335, winog: 0.645, piqa: 0.644, hellas: 0.521, obqa: 0.334, mcq: 0.509, mathqa: 0.238, mmlum: 0.275, qret: 0.690 }
      ]
    },
    "llama-31-8b": {
      label: "Llama-3.1-8B",
      rows: [
        { keep: "100%", method: "Dense FP", wt2: 6.24, c4: 9.10, boolq: 0.831, arce: 0.824, arcc: 0.549, winog: 0.746, piqa: 0.812, hellas: 0.793, obqa: 0.454, mcq: 0.716, mathqa: 0.396, mmlum: 0.437, qret: 1.000 },
        { keep: "80%", method: "ASVD", wt2: 2011.38, c4: 1281.96, boolq: 0.382, arce: 0.285, arcc: 0.226, winog: 0.512, piqa: 0.536, hellas: 0.285, obqa: 0.244, mcq: 0.353, mathqa: 0.201, mmlum: 0.223, qret: 0.304 },
        { keep: "80%", method: "SVD-LLM v1", wt2: 14.83, c4: 80.94, boolq: 0.661, arce: 0.528, arcc: 0.315, winog: 0.645, piqa: 0.639, hellas: 0.476, obqa: 0.350, mcq: 0.516, mathqa: 0.256, mmlum: 0.305, qret: 0.520 },
        { keep: "80%", method: "DoBi-SVD", wt2: 556.59, c4: 1008.41, boolq: 0.378, arce: 0.298, arcc: 0.226, winog: 0.516, piqa: 0.522, hellas: 0.282, obqa: 0.266, mcq: 0.355, mathqa: 0.205, mmlum: 0.292, qret: 0.341 },
        { keep: "80%", method: "Basis Sharing", wt2: 15.61, c4: 54.36, boolq: 0.632, arce: 0.637, arcc: 0.367, winog: 0.667, piqa: 0.701, hellas: 0.548, obqa: 0.372, mcq: 0.561, mathqa: 0.248, mmlum: 0.297, qret: 0.531 },
        { keep: "80%", method: "MoDeGPT", wt2: 9.01, c4: 17.68, boolq: 0.412, arce: 0.715, arcc: 0.436, winog: 0.730, piqa: 0.743, hellas: 0.710, obqa: 0.382, mcq: 0.590, mathqa: 0.344, mmlum: 0.407, qret: 0.766 },
        { keep: "60%", method: "ASVD", wt2: 22684.63, c4: 14186.23, boolq: 0.405, arce: 0.254, arcc: 0.257, winog: 0.491, piqa: 0.507, hellas: 0.260, obqa: 0.284, mcq: 0.351, mathqa: 0.192, mmlum: 0.286, qret: 0.326 },
        { keep: "60%", method: "SVD-LLM v1", wt2: 199.84, c4: 1187.78, boolq: 0.378, arce: 0.295, arcc: 0.246, winog: 0.533, piqa: 0.515, hellas: 0.283, obqa: 0.268, mcq: 0.360, mathqa: 0.205, mmlum: 0.257, qret: 0.329 },
        { keep: "60%", method: "DoBi-SVD", wt2: 987.51, c4: 1529.38, boolq: 0.378, arce: 0.271, arcc: 0.251, winog: 0.481, piqa: 0.511, hellas: 0.265, obqa: 0.288, mcq: 0.349, mathqa: 0.203, mmlum: 0.268, qret: 0.325 },
        { keep: "60%", method: "Basis Sharing", wt2: 82.96, c4: 461.21, boolq: 0.380, arce: 0.409, arcc: 0.241, winog: 0.562, piqa: 0.568, hellas: 0.325, obqa: 0.284, mcq: 0.396, mathqa: 0.205, mmlum: 0.211, qret: 0.330 },
        { keep: "60%", method: "MoDeGPT", wt2: 24.50, c4: 51.82, boolq: 0.622, arce: 0.460, arcc: 0.312, winog: 0.672, piqa: 0.629, hellas: 0.516, obqa: 0.316, mcq: 0.504, mathqa: 0.241, mmlum: 0.213, qret: 0.446 }
      ]
    },
    "qwen3-8b": {
      label: "Qwen3-8B-Base",
      rows: [
        { keep: "100%", method: "Dense FP", wt2: 7.00, c4: 11.78, boolq: 0.830, arce: 0.800, arcc: 0.570, winog: 0.727, piqa: 0.793, hellas: 0.787, obqa: 0.420, mcq: 0.704, mathqa: 0.542, mmlum: 0.729, qret: 1.000 },
        { keep: "80%", method: "ASVD", wt2: 11.88, c4: 20.54, boolq: 0.792, arce: 0.758, arcc: 0.483, winog: 0.651, piqa: 0.745, hellas: 0.641, obqa: 0.414, mcq: 0.641, mathqa: 0.420, mmlum: 0.460, qret: 0.696 },
        { keep: "80%", method: "SVD-LLM v1", wt2: 11.08, c4: 33.90, boolq: 0.688, arce: 0.649, arcc: 0.424, winog: 0.669, piqa: 0.712, hellas: 0.616, obqa: 0.408, mcq: 0.595, mathqa: 0.330, mmlum: 0.355, qret: 0.584 },
        { keep: "80%", method: "DoBi-SVD", wt2: 51.21, c4: 236.65, boolq: 0.460, arce: 0.335, arcc: 0.284, winog: 0.519, piqa: 0.550, hellas: 0.388, obqa: 0.262, mcq: 0.400, mathqa: 0.206, mmlum: 0.264, qret: 0.299 },
        { keep: "80%", method: "Basis Sharing", wt2: 11.05, c4: 31.67, boolq: 0.676, arce: 0.578, arcc: 0.439, winog: 0.673, piqa: 0.726, hellas: 0.635, obqa: 0.388, mcq: 0.588, mathqa: 0.331, mmlum: 0.347, qret: 0.585 },
        { keep: "80%", method: "MoDeGPT", wt2: 10.34, c4: 22.46, boolq: 0.658, arce: 0.613, arcc: 0.430, winog: 0.655, piqa: 0.731, hellas: 0.683, obqa: 0.408, mcq: 0.597, mathqa: 0.279, mmlum: 0.297, qret: 0.594 },
        { keep: "60%", method: "ASVD", wt2: 1359.38, c4: 1484.57, boolq: 0.502, arce: 0.292, arcc: 0.241, winog: 0.499, piqa: 0.544, hellas: 0.281, obqa: 0.270, mcq: 0.375, mathqa: 0.212, mmlum: 0.234, qret: 0.252 },
        { keep: "60%", method: "SVD-LLM v1", wt2: 20.44, c4: 112.01, boolq: 0.544, arce: 0.382, arcc: 0.265, winog: 0.545, piqa: 0.590, hellas: 0.385, obqa: 0.266, mcq: 0.425, mathqa: 0.231, mmlum: 0.307, qret: 0.380 },
        { keep: "60%", method: "DoBi-SVD", wt2: 107.63, c4: 610.21, boolq: 0.511, arce: 0.290, arcc: 0.253, winog: 0.511, piqa: 0.523, hellas: 0.310, obqa: 0.306, mcq: 0.386, mathqa: 0.203, mmlum: 0.302, qret: 0.284 },
        { keep: "60%", method: "Basis Sharing", wt2: 18.59, c4: 95.39, boolq: 0.630, arce: 0.457, arcc: 0.278, winog: 0.580, piqa: 0.620, hellas: 0.423, obqa: 0.298, mcq: 0.469, mathqa: 0.253, mmlum: 0.302, qret: 0.409 },
        { keep: "60%", method: "MoDeGPT", wt2: 18.40, c4: 55.66, boolq: 0.509, arce: 0.414, arcc: 0.293, winog: 0.569, piqa: 0.625, hellas: 0.460, obqa: 0.328, mcq: 0.457, mathqa: 0.228, mmlum: 0.246, qret: 0.400 }
      ]
    }
  },
  compressionComparison: {
    ratios: ["80%", "70%", "60%", "50%", "40%"],
    metrics: {
      mcq: {
        label: "MCQ Avg. ↑",
        direction: "max",
        decimals: 3,
        rows: [
          { category: "Full precision", method: "Dense Model", training: "None", values: [0.715, 0.715, 0.715, 0.715, 0.715], dense: true },
          { category: "Structured pruning", method: "LLM-Pruner", training: "Calibration", values: [0.625, 0.551, 0.479, 0.446, 0.431] },
          { category: "Structured pruning", method: "SliceGPT", training: "Calibration", values: [0.527, 0.443, 0.386, 0.359, 0.350] },
          { category: "Structured pruning", method: "BlockPruner", training: "Calibration", values: [0.570, 0.501, 0.424, 0.365, 0.351] },
          { category: "Low-rank", method: "SVD-LLM v1", training: "Calibration", values: [0.516, 0.401, 0.360, 0.343, 0.346] },
          { category: "Low-rank", method: "Basis Sharing", training: "Calibration", values: [0.561, 0.488, 0.395, 0.359, 0.350] },
          { category: "Low-rank", method: "MoDeGPT", training: "Calibration", values: [0.590, 0.567, 0.504, 0.439, 0.381] }
        ]
      },
      c4: {
        label: "C4 PPL ↓",
        direction: "min",
        decimals: 2,
        rows: [
          { category: "Full precision", method: "Dense Model", training: "None", values: [9.10, 9.10, 9.10, 9.10, 9.10], dense: true },
          { category: "Structured pruning", method: "LLM-Pruner", training: "Calibration", values: [14.53, 20.78, 34.85, 55.20, 75.34] },
          { category: "Structured pruning", method: "SliceGPT", training: "Calibration", values: [117.30, 220.46, 412.76, 610.05, 885.63] },
          { category: "Structured pruning", method: "BlockPruner", training: "Calibration", values: [21.68, 39.96, 120.65, 600.88, 6766.48] },
          { category: "Low-rank", method: "SVD-LLM v1", training: "Calibration", values: [80.94, 298.59, 1187.78, 18935.68, 64335.93] },
          { category: "Low-rank", method: "Basis Sharing", training: "Calibration", values: [54.36, 155.46, 461.21, 1194.51, 2354.98] },
          { category: "Low-rank", method: "MoDeGPT", training: "Calibration", values: [17.68, 26.19, 51.82, 137.87, 502.80] }
        ]
      }
    }
  },
  serving: {
    profiles: {
      prefill: { label: "Prefill-heavy", tokens: "4096 → 32" },
      balanced: { label: "Balanced", tokens: "2048 → 128" },
      decode: { label: "Decode-heavy", tokens: "512 → 512" }
    },
    rows: [
      { method: "Dense BF16", group: "reference", prefill: [1033.71, 3507.69, 11521.19, 90.03], balanced: [73.00, 1730.49, 10896.70, 681.38], decode: [51.19, 6101.17, 769.37, 770.88] },
      { method: "ASVD", group: "svd", prefill: [906.88, 3602.72, 11154.51, 87.17], balanced: [93.72, 2981.86, 6328.40, 395.72], decode: [72.32, 11053.07, 423.96, 424.79] },
      { method: "SVD-LLM v1", group: "svd", prefill: [976.19, 3852.98, 10439.98, 81.58], balanced: [94.42, 3031.04, 6232.10, 389.70], decode: [73.24, 11251.25, 416.68, 417.49] },
      { method: "SVD-LLM v2", group: "svd", prefill: [967.90, 3831.87, 10490.16, 81.97], balanced: [101.52, 3101.10, 6058.19, 378.82], decode: [75.84, 11557.80, 405.04, 405.83] },
      { method: "DoBi-SVD", group: "svd", prefill: [1011.50, 3943.80, 10222.12, 79.88], balanced: [94.96, 2712.51, 6957.39, 435.05], decode: [69.85, 10040.88, 467.48, 468.39] },
      { method: "Basis Sharing", group: "svd", prefill: [1092.77, 4295.30, 9372.91, 73.24], balanced: [111.55, 3278.81, 5747.93, 359.42], decode: [79.05, 12253.39, 383.41, 384.16] },
      { method: "LLM-Pruner†", group: "audit", prefill: [368.71, 730.42, 6643.60, 51.92], balanced: [24.63, 1506.62, 1358.97, 84.98], decode: [21.98, 6012.41, 84.99, 85.15] },
      { method: "SliceGPT†", group: "audit", prefill: [320.05, 729.26, 6477.84, 50.62], balanced: [24.52, 1692.76, 1209.10, 75.61], decode: [22.08, 6763.40, 75.57, 75.72] }
    ]
  }
};
