(function () {
  const slides = Array.from(document.querySelectorAll(".slide"));
  const prev = document.getElementById("prev-slide");
  const next = document.getElementById("next-slide");
  const count = document.getElementById("slide-count");
  const progress = document.getElementById("progress-bar");

  let current = 0;

  function clampSlide(index) {
    return Math.max(0, Math.min(index, slides.length - 1));
  }

  function render(index) {
    current = clampSlide(index);
    slides.forEach((slide, slideIndex) => {
      slide.classList.toggle("active", slideIndex === current);
      slide.setAttribute("aria-hidden", slideIndex === current ? "false" : "true");
    });

    count.textContent = `${current + 1} / ${slides.length}`;
    progress.style.width = `${((current + 1) / slides.length) * 100}%`;
    prev.disabled = current === 0;
    next.disabled = current === slides.length - 1;

    const targetHash = `#${current + 1}`;
    if (window.location.hash !== targetHash) {
      history.replaceState(null, "", targetHash);
    }
  }

  function fromHash() {
    const value = Number.parseInt(window.location.hash.replace("#", ""), 10);
    if (Number.isFinite(value)) {
      return value - 1;
    }
    return 0;
  }

  prev.addEventListener("click", () => render(current - 1));
  next.addEventListener("click", () => render(current + 1));

  window.addEventListener("keydown", (event) => {
    const key = event.key;
    if (["ArrowRight", "PageDown", " "].includes(key)) {
      event.preventDefault();
      render(current + 1);
    }
    if (["ArrowLeft", "PageUp", "Backspace"].includes(key)) {
      event.preventDefault();
      render(current - 1);
    }
    if (key === "Home") {
      event.preventDefault();
      render(0);
    }
    if (key === "End") {
      event.preventDefault();
      render(slides.length - 1);
    }
  });

  window.addEventListener("hashchange", () => render(fromHash()));

  render(fromHash());
})();
