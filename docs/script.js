// Populate the PC Box grid from the real 151-entry class list.
(function buildBoxGrid() {
  const grid = document.getElementById("boxGrid");
  const frag = document.createDocumentFragment();

  POKEMON_CLASSES.forEach(({ id, name }) => {
    const cell = document.createElement("div");
    cell.className = "box-cell";
    cell.dataset.id = id;
    cell.dataset.name = name;

    const well = document.createElement("div");
    well.className = "box-cell-well";

    const img = document.createElement("img");
    img.src = spriteUrl(id);
    img.alt = name;
    img.loading = "lazy";
    well.appendChild(img);

    const label = document.createElement("div");
    label.className = "box-cell-label";
    label.textContent = name;

    const dex = document.createElement("div");
    dex.className = "box-cell-dex";
    dex.textContent = "#" + String(id).padStart(3, "0");

    cell.appendChild(well);
    cell.appendChild(label);
    cell.appendChild(dex);
    frag.appendChild(cell);
  });

  grid.appendChild(frag);
})();

// Floating sprite preview: hovering a box cell shows a larger view of that
// Pokémon's sprite in a small window that follows the cursor.
(function spriteHoverPreview() {
  const grid = document.getElementById("boxGrid");
  const preview = document.createElement("div");
  preview.className = "sprite-preview";
  const img = document.createElement("img");
  preview.appendChild(img);
  document.body.appendChild(preview);

  const SIZE = 130;
  let activeId = null;

  grid.addEventListener("mousemove", (e) => {
    const cell = e.target.closest(".box-cell");
    if (!cell) {
      preview.classList.remove("is-visible");
      activeId = null;
      return;
    }

    const id = cell.dataset.id;
    if (id !== activeId) {
      img.src = spriteUrl(id);
      img.alt = cell.dataset.name || "";
      activeId = id;
    }

    let x = e.clientX + 18;
    let y = e.clientY + 18;
    if (x + SIZE > window.innerWidth) x = e.clientX - SIZE - 18;
    if (y + SIZE > window.innerHeight) y = e.clientY - SIZE - 18;
    preview.style.left = x + "px";
    preview.style.top = y + "px";
    preview.classList.add("is-visible");
  });

  grid.addEventListener("mouseleave", () => {
    preview.classList.remove("is-visible");
    activeId = null;
  });
})();

// Upload zone: drag & drop or click-to-browse, shows the chosen photo.
// Display only — no model wired in yet, so this doesn't run any inference.
(function uploadPreview() {
  const zone = document.getElementById("uploadZone");
  const input = document.getElementById("fileInput");
  const well = document.getElementById("originalWell");

  function showFile(file) {
    if (!file || !file.type || !file.type.startsWith("image/")) return;
    const img = document.createElement("img");
    img.alt = "Photo you uploaded";
    img.onload = () => handleUploadedImage(img);
    img.src = URL.createObjectURL(file);
    well.innerHTML = "";
    well.appendChild(img);
  }

  input.addEventListener("change", () => {
    if (input.files && input.files[0]) showFile(input.files[0]);
  });

  ["dragover", "dragenter"].forEach((evt) =>
    zone.addEventListener(evt, (e) => {
      e.preventDefault();
      zone.classList.add("is-dragover");
    })
  );
  ["dragleave", "dragend"].forEach((evt) =>
    zone.addEventListener(evt, () => zone.classList.remove("is-dragover"))
  );
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("is-dragover");
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) showFile(file);
  });
})();

// Theme toggle: defaults to system preference, remembers an explicit choice.
(function themeToggle() {
  const root = document.documentElement;
  const button = document.getElementById("themeToggle");
  const icon = document.getElementById("themeIcon");
  const label = document.getElementById("themeLabel");
  const STORAGE_KEY = "pokedex-site-theme";

  function apply(theme) {
    if (theme) {
      root.setAttribute("data-theme", theme);
    } else {
      root.removeAttribute("data-theme");
    }
    const isDark = theme === "dark" ||
      (!theme && window.matchMedia("(prefers-color-scheme: dark)").matches);
    button.setAttribute("aria-pressed", String(isDark));
    icon.textContent = isDark ? "☀" : "☽"; // sun / moon
    label.textContent = isDark ? "Day Mode" : "Night Mode";
  }

  let stored = null;
  try {
    stored = localStorage.getItem(STORAGE_KEY);
  } catch (e) {
    // Storage unavailable (private mode, etc.) — fall back to system theme.
  }
  apply(stored);

  button.addEventListener("click", () => {
    const currentlyDark = root.getAttribute("data-theme") === "dark" ||
      (!root.getAttribute("data-theme") && window.matchMedia("(prefers-color-scheme: dark)").matches);
    const next = currentlyDark ? "light" : "dark";
    apply(next);
    try {
      localStorage.setItem(STORAGE_KEY, next);
    } catch (e) {
      // Ignore write failures — theme just won't persist this session.
    }
  });
})();
