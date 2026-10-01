/* Shared interface behaviour. One definition, so every page behaves the same.
   Light/dark is a selected pair of steps, not an inversion — see shared/ui.css. */
(function(){
  const sysDark = window.matchMedia("(prefers-color-scheme: dark)");
  let stored = null;
  try { stored = localStorage.getItem("ui-theme"); } catch (e) {}
  if (stored) document.documentElement.setAttribute("data-theme", stored);

  const isDark = () => (document.documentElement.getAttribute("data-theme")
    || (sysDark.matches ? "dark" : "light")) === "dark";

  function wire(btn){
    if (!btn) return;
    const sync = () => { btn.textContent = isDark() ? "Light" : "Dark"; };
    btn.addEventListener("click", () => {
      const next = isDark() ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      try { localStorage.setItem("ui-theme", next); } catch (e) {}
      sync();
      dispatchEvent(new CustomEvent("themechange", {detail: {dark: next === "dark"}}));
    });
    sysDark.addEventListener("change", sync);
    sync();
  }

  const start = () => wire(document.getElementById("themebtn"));
  if (document.readyState === "loading")
    document.addEventListener("DOMContentLoaded", start);
  else start();
})();
