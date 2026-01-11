/* global window */
// Module for managing prompt term lists

// State
let positiveTerms = [];
let negativeTerms = [];
let maskTerms = {
  must_include: [],
  ban_terms: [],
  avoid_terms: []
};

// DOM elements
function getElements() {
  return {
    positiveList: document.getElementById("positiveTermsList"),
    negativeList: document.getElementById("negativeTermsList"),
    mustIncludeList: document.getElementById("mustIncludeTermsList"),
    banTermsList: document.getElementById("banTermsList"),
    avoidTermsList: document.getElementById("avoidTermsList"),
    positiveInput: document.getElementById("addPositiveTerm"),
    negativeInput: document.getElementById("addNegativeTerm"),
    mustIncludeInput: document.getElementById("addMustIncludeTerm"),
    banTermsInput: document.getElementById("addBanTerm"),
    avoidTermsInput: document.getElementById("addAvoidTerm")
  };
}

// Render a term list with drag-and-drop support
function renderTermList(container, terms, onRemove, type = "default", onReorder = null) {
  if (!container) return;
  
  container.innerHTML = "";
  
  if (terms.length === 0) {
    container.innerHTML = '<li class="has-text-grey is-size-7">No terms</li>';
    return;
  }
  
  terms.forEach((term, idx) => {
    const li = document.createElement("li");
    li.className = "term-item";
    li.draggable = true;
    li.dataset.index = idx;
    li.innerHTML = `
      <span class="icon is-small has-text-grey" style="cursor: move; margin-right: 0.5rem;">
        <i class="fas fa-grip-vertical"></i>
      </span>
      <span class="term-text">${escapeHtml(term)}</span>
      <button class="button is-small is-danger is-outlined delete-term" data-index="${idx}" data-type="${type}">
        <span class="icon is-small"><i class="fas fa-trash"></i></span>
      </button>
    `;
    container.appendChild(li);
  });
  
  // Wire delete buttons
  container.querySelectorAll(".delete-term").forEach(btn => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const idx = parseInt(btn.dataset.index);
      onRemove(idx);
    });
  });
  
  // Wire drag-and-drop if onReorder is provided
  if (onReorder) {
    let draggedElement = null;
    let draggedIndex = null;
    
    container.querySelectorAll(".term-item").forEach((item, idx) => {
      item.addEventListener("dragstart", (e) => {
        draggedElement = item;
        draggedIndex = idx;
        item.classList.add("dragging");
        e.dataTransfer.effectAllowed = "move";
        e.dataTransfer.setData("text/html", item.innerHTML);
      });
      
      item.addEventListener("dragend", () => {
        item.classList.remove("dragging");
        container.querySelectorAll(".term-item").forEach(i => i.classList.remove("drag-over"));
      });
      
      item.addEventListener("dragover", (e) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = "move";
        
        const afterElement = getDragAfterElement(container, e.clientY);
        container.querySelectorAll(".term-item").forEach(i => i.classList.remove("drag-over"));
        if (afterElement == null) {
          item.classList.add("drag-over");
        } else {
          afterElement.classList.add("drag-over");
        }
      });
      
      item.addEventListener("drop", (e) => {
        e.preventDefault();
        if (draggedElement && draggedIndex !== null) {
          const afterElement = getDragAfterElement(container, e.clientY);
          const newIndex = afterElement ? parseInt(afterElement.dataset.index) : terms.length - 1;
          
          if (draggedIndex !== newIndex) {
            onReorder(draggedIndex, newIndex);
          }
        }
        container.querySelectorAll(".term-item").forEach(i => i.classList.remove("drag-over"));
      });
    });
  }
}

// Helper to find element after drag position
function getDragAfterElement(container, y) {
  const draggableElements = [...container.querySelectorAll(".term-item:not(.dragging)")];
  
  return draggableElements.reduce((closest, child) => {
    const box = child.getBoundingClientRect();
    const offset = y - box.top - box.height / 2;
    
    if (offset < 0 && offset > closest.offset) {
      return { offset: offset, element: child };
    } else {
      return closest;
    }
  }, { offset: Number.NEGATIVE_INFINITY }).element;
}

// Add a term
function addTerm(list, term, type) {
  const trimmed = term.trim();
  if (!trimmed) return false;
  
  // Check for duplicates (case-insensitive)
  const lower = trimmed.toLowerCase();
  if (list.some(t => t.toLowerCase() === lower)) {
    return false; // Duplicate
  }
  
  list.push(trimmed);
  return true;
}

// Remove a term
function removeTerm(list, index) {
  if (index >= 0 && index < list.length) {
    list.splice(index, 1);
    return true;
  }
  return false;
}

// Initialize prompt lists
function initPrompts() {
  const els = getElements();
  if (!els.positiveList || !els.negativeList) return;
  
  // Wire add buttons
  if (els.positiveInput) {
    els.positiveInput.addEventListener("keypress", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        if (addTerm(positiveTerms, els.positiveInput.value, "positive")) {
          els.positiveInput.value = "";
          renderPositiveTerms();
          if (window.runValidation) {
            window.runValidation();
          }
        }
      }
    });
  }
  
  if (els.negativeInput) {
    els.negativeInput.addEventListener("keypress", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        if (addTerm(negativeTerms, els.negativeInput.value, "negative")) {
          els.negativeInput.value = "";
          renderNegativeTerms();
          if (window.runValidation) {
            window.runValidation();
          }
        }
      }
    });
  }
  
  // Wire mask term inputs
  if (els.mustIncludeInput) {
    els.mustIncludeInput.addEventListener("keypress", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        if (addTerm(maskTerms.must_include, els.mustIncludeInput.value, "must_include")) {
          els.mustIncludeInput.value = "";
          renderMaskTerms();
        }
      }
    });
  }
  
  if (els.banTermsInput) {
    els.banTermsInput.addEventListener("keypress", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        if (addTerm(maskTerms.ban_terms, els.banTermsInput.value, "ban_terms")) {
          els.banTermsInput.value = "";
          renderMaskTerms();
        }
      }
    });
  }
  
  if (els.avoidTermsInput) {
    els.avoidTermsInput.addEventListener("keypress", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        if (addTerm(maskTerms.avoid_terms, els.avoidTermsInput.value, "avoid_terms")) {
          els.avoidTermsInput.value = "";
          renderMaskTerms();
        }
      }
    });
  }
}

// Reorder terms
function reorderTerms(list, fromIndex, toIndex) {
  if (fromIndex === toIndex) return;
  const [removed] = list.splice(fromIndex, 1);
  list.splice(toIndex, 0, removed);
}

// Render functions
function renderPositiveTerms() {
  const els = getElements();
  renderTermList(els.positiveList, positiveTerms, (idx) => {
    removeTerm(positiveTerms, idx);
    renderPositiveTerms();
    // Trigger validation after render
    if (window.runValidation) {
      window.runValidation();
    }
  }, "positive", (fromIdx, toIdx) => {
    reorderTerms(positiveTerms, fromIdx, toIdx);
    renderPositiveTerms();
  });
  // Trigger validation after initial render too
  if (window.runValidation) {
    window.runValidation();
  }
}

function renderNegativeTerms() {
  const els = getElements();
  renderTermList(els.negativeList, negativeTerms, (idx) => {
    removeTerm(negativeTerms, idx);
    renderNegativeTerms();
    // Trigger validation after render
    if (window.runValidation) {
      window.runValidation();
    }
  }, "negative", (fromIdx, toIdx) => {
    reorderTerms(negativeTerms, fromIdx, toIdx);
    renderNegativeTerms();
  });
  // Trigger validation after initial render too
  if (window.runValidation) {
    window.runValidation();
  }
}

function renderMaskTerms() {
  const els = getElements();
  if (els.mustIncludeList) {
    renderTermList(els.mustIncludeList, maskTerms.must_include, (idx) => {
      removeTerm(maskTerms.must_include, idx);
      renderMaskTerms();
    }, "must_include", (fromIdx, toIdx) => {
      reorderTerms(maskTerms.must_include, fromIdx, toIdx);
      renderMaskTerms();
    });
  }
  if (els.banTermsList) {
    renderTermList(els.banTermsList, maskTerms.ban_terms, (idx) => {
      removeTerm(maskTerms.ban_terms, idx);
      renderMaskTerms();
    }, "ban_terms", (fromIdx, toIdx) => {
      reorderTerms(maskTerms.ban_terms, fromIdx, toIdx);
      renderMaskTerms();
    });
  }
  if (els.avoidTermsList) {
    renderTermList(els.avoidTermsList, maskTerms.avoid_terms, (idx) => {
      removeTerm(maskTerms.avoid_terms, idx);
      renderMaskTerms();
    }, "avoid_terms", (fromIdx, toIdx) => {
      reorderTerms(maskTerms.avoid_terms, fromIdx, toIdx);
      renderMaskTerms();
    });
  }
}

// Load terms from API
async function loadTermsFromAPI() {
  // This will be called from the main live.js
  // For now, just initialize
  initPrompts();
}

// Convert term lists to prompt strings
function termsToPrompt(terms) {
  return terms.join(", ");
}

// Parse prompt string to term list
function promptToTerms(promptStr) {
  if (!promptStr) return [];
  return promptStr.split(",").map(t => t.trim()).filter(t => t);
}

// Get current prompts as strings
function getCurrentPrompts() {
  return {
    positive: termsToPrompt(positiveTerms),
    negative: termsToPrompt(negativeTerms)
  };
}

// Set terms from prompt strings
function setTermsFromPrompts(positive, negative) {
  positiveTerms = promptToTerms(positive);
  negativeTerms = promptToTerms(negative);
  renderPositiveTerms();
  renderNegativeTerms();
}

// Set mask terms
function setMaskTerms(must_include, ban_terms, avoid_terms) {
  maskTerms.must_include = Array.isArray(must_include) ? must_include : [];
  maskTerms.ban_terms = Array.isArray(ban_terms) ? ban_terms : [];
  maskTerms.avoid_terms = Array.isArray(avoid_terms) ? avoid_terms : [];
  renderMaskTerms();
}

// Get mask terms
function getMaskTerms() {
  return { ...maskTerms };
}

// Utility
function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

// Export
window.PromptsModule = {
  init: initPrompts,
  loadTermsFromAPI,
  renderPositiveTerms,
  renderNegativeTerms,
  renderMaskTerms,
  getCurrentPrompts,
  setTermsFromPrompts,
  setMaskTerms,
  getMaskTerms,
  termsToPrompt,
  promptToTerms,
  // Expose internal state for validation
  getPositiveTerms: () => positiveTerms,
  getNegativeTerms: () => negativeTerms
};
