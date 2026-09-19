// Preprocessing that mirrors model.py's eval transform exactly:
//   transforms.Resize(eval_resize)      -> shorter side to 288px, aspect ratio kept
//   transforms.CenterCrop(eval_crop)    -> centre 256x256
//   transforms.ToTensor()               -> [0,255] uint8 -> [0,1] float, HWC -> CHW
//   transforms.Normalize(norm_mean, norm_std)
// Values below match model.py's module-level constants of the same names.
const NORM_MEAN = [0.485, 0.456, 0.406];
const NORM_STD = [0.229, 0.224, 0.225];
const EVAL_RESIZE = 288;
const EVAL_CROP = 256;

// Takes a loaded HTMLImageElement, returns a Float32Array in NCHW layout
// ([1, 3, 256, 256]), normalised the same way the model was trained/tested.
function preprocessImage(img) {
  // 1. Resize so the shorter side is EVAL_RESIZE, preserving aspect ratio.
  const scale = EVAL_RESIZE / Math.min(img.naturalWidth, img.naturalHeight);
  const resizedW = Math.round(img.naturalWidth * scale);
  const resizedH = Math.round(img.naturalHeight * scale);

  const resizeCanvas = document.createElement("canvas");
  resizeCanvas.width = resizedW;
  resizeCanvas.height = resizedH;
  const resizeCtx = resizeCanvas.getContext("2d");
  resizeCtx.drawImage(img, 0, 0, resizedW, resizedH);

  // 2. Centre-crop to EVAL_CROP x EVAL_CROP.
  const cropX = Math.round((resizedW - EVAL_CROP) / 2);
  const cropY = Math.round((resizedH - EVAL_CROP) / 2);

  const cropCanvas = document.createElement("canvas");
  cropCanvas.width = EVAL_CROP;
  cropCanvas.height = EVAL_CROP;
  const cropCtx = cropCanvas.getContext("2d");
  cropCtx.drawImage(
    resizeCanvas,
    cropX, cropY, EVAL_CROP, EVAL_CROP,
    0, 0, EVAL_CROP, EVAL_CROP
  );

  // 3. Pixels -> normalised Float32Array, HWC (RGBA) -> CHW (RGB only).
  const { data } = cropCtx.getImageData(0, 0, EVAL_CROP, EVAL_CROP);
  const chwSize = EVAL_CROP * EVAL_CROP;
  const output = new Float32Array(3 * chwSize);

  for (let i = 0; i < chwSize; i++) {
    const r = data[i * 4] / 255;
    const g = data[i * 4 + 1] / 255;
    const b = data[i * 4 + 2] / 255;
    output[i] = (r - NORM_MEAN[0]) / NORM_STD[0];               // R plane
    output[chwSize + i] = (g - NORM_MEAN[1]) / NORM_STD[1];     // G plane
    output[2 * chwSize + i] = (b - NORM_MEAN[2]) / NORM_STD[2]; // B plane
  }

  return output; // shape [1, 3, 256, 256] once wrapped in an ort.Tensor
}

// --- Class-index mapping -----------------------------------------------
// The model's output indices follow train_classes.json (the exact
// alphabetical folder order ImageFolder used during training) — NOT the
// Dex-ordered, prettily-punctuated names in classes.js. Match the two by
// normalising both to a bare lowercase alphanumeric string, special-casing
// the gender symbols (Nidoran's the only case that needs it).
function normalizeName(name) {
  return name
    .replace(/♀/g, "f")
    .replace(/♂/g, "m")
    .toLowerCase()
    .replace(/[^a-z0-9]/g, "");
}

// --- Model + class mapping, loaded once and kicked off immediately on ---
// --- page load (not on upload) so they're ready by the time someone   ---
// --- picks a photo, instead of adding their download time to the wait ---
let sessionPromise = null;
function loadModel() {
  if (!sessionPromise) {
    sessionPromise = ort.InferenceSession.create("model/pokedex.onnx");
  }
  return sessionPromise;
}

let indexToPokemonPromise = null;
function loadClassMapping() {
  if (!indexToPokemonPromise) {
    indexToPokemonPromise = fetch("model/train_classes.json")
      .then((r) => r.json())
      .then((trainClasses) => {
        const byNormalizedName = new Map(
          POKEMON_CLASSES.map((p) => [normalizeName(p.name), p])
        );
        return trainClasses.map((name) => {
          const match = byNormalizedName.get(normalizeName(name));
          if (!match) console.warn("No class-list match for", name);
          return match;
        });
      });
  }
  return indexToPokemonPromise;
}

loadModel();
loadClassMapping();

// --- Softmax: converts the model's raw logits into percentages --------
function softmax(logits) {
  const max = Math.max(...logits);
  const exps = Array.from(logits, (x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((x) => x / sum);
}

// Below this confidence, treat the upload as not one of the 151 rather
// than reporting a low-confidence guess. Untuned starting point — revisit
// once there's a feel for how confident real vs. off-target photos score.
const CONFIDENCE_THRESHOLD = 0.4;

// --- Result display -----------------------------------------------------
function setPreviewSprite(url, alt) {
  const well = document.getElementById("previewWell");
  well.innerHTML = "";
  const img = document.createElement("img");
  img.className = "preview-sprite";
  img.src = url;
  img.alt = alt;
  well.appendChild(img);
}

function clearPreviewSprite(message) {
  const well = document.getElementById("previewWell");
  well.innerHTML = "";
  const span = document.createElement("span");
  span.className = "preview-placeholder";
  span.textContent = message;
  well.appendChild(span);
}

function setResultNote(text) {
  const note = document.getElementById("resultNote");
  if (text) {
    note.textContent = text;
    note.hidden = false;
  } else {
    note.hidden = true;
  }
}

function showLoadingState() {
  clearPreviewSprite("Identifying…");
  document.getElementById("pokeName").textContent = "Identifying…";
  document.getElementById("dexValue").textContent = "—";
  document.getElementById("typeValue").textContent = "—";
  document.getElementById("confidenceValue").textContent = "—";
  setResultNote(null);
}

function showMatchResult(pokemon, confidence) {
  setPreviewSprite(spriteUrl(pokemon.id), pokemon.name);
  document.getElementById("pokeName").textContent = pokemon.name;
  document.getElementById("dexValue").textContent = "#" + String(pokemon.id).padStart(3, "0");
  document.getElementById("typeValue").textContent = pokemon.type;
  document.getElementById("confidenceValue").textContent = (confidence * 100).toFixed(1) + "%";
  setResultNote(null);
}

function showNoMatchResult(confidence) {
  clearPreviewSprite("No match");
  document.getElementById("pokeName").textContent = "Not one of the original 151?";
  document.getElementById("dexValue").textContent = "—";
  document.getElementById("typeValue").textContent = "—";
  document.getElementById("confidenceValue").textContent = (confidence * 100).toFixed(1) + "%";
  setResultNote("Pokemon doesn't seem to be one of the original 151.");
}

function showErrorState(message) {
  clearPreviewSprite("Error");
  document.getElementById("pokeName").textContent = "Something went wrong";
  document.getElementById("dexValue").textContent = "—";
  document.getElementById("typeValue").textContent = "—";
  document.getElementById("confidenceValue").textContent = "—";
  setResultNote(message);
}

// --- Entry point: called from script.js once an uploaded photo has loaded
async function handleUploadedImage(img) {
  showLoadingState();
  try {
    const [session, indexToPokemon] = await Promise.all([loadModel(), loadClassMapping()]);

    const inputData = preprocessImage(img);
    const tensor = new ort.Tensor("float32", inputData, [1, 3, EVAL_CROP, EVAL_CROP]);
    const outputs = await session.run({ input: tensor });
    const logits = outputs.logits.data;
    const probs = softmax(logits);

    let bestIdx = 0;
    for (let i = 1; i < probs.length; i++) {
      if (probs[i] > probs[bestIdx]) bestIdx = i;
    }
    const confidence = probs[bestIdx];
    const pokemon = indexToPokemon[bestIdx];

    if (!pokemon || confidence < CONFIDENCE_THRESHOLD) {
      showNoMatchResult(confidence);
    } else {
      showMatchResult(pokemon, confidence);
    }
  } catch (err) {
    console.error(err);
    showErrorState("Couldn't identify that photo — try a different one.");
  }
}
