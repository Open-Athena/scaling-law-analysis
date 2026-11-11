const copyButton = document.getElementById("copy-bibtex");
const citation = document.getElementById("bibtex");
const copyStatus = document.getElementById("copy-status");
let resetCopyLabel;

copyButton.hidden = false;
copyButton.addEventListener("click", async () => {
  clearTimeout(resetCopyLabel);
  try {
    await navigator.clipboard.writeText(citation.textContent);
    copyButton.textContent = "Copied!";
    copyStatus.textContent = "BibTeX citation copied to clipboard.";
  } catch {
    const range = document.createRange();
    range.selectNodeContents(citation);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    copyButton.textContent = "Citation selected";
    copyStatus.textContent = "Citation selected. Use your keyboard to copy.";
  }
  resetCopyLabel = setTimeout(() => {
    copyButton.textContent = "Copy BibTeX";
    copyStatus.textContent = "";
  }, 2500);
});
