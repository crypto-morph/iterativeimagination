/* global window */
// Module for handling AIVis suggestions

let currentSuggestions = null;

// Render suggestions panel
function renderSuggestions(suggestions, onApply) {
  const container = document.getElementById("suggestionsPanel");
  if (!container) {
    console.error("suggestionsPanel element not found");
    return;
  }
  
  if (!suggestions) {
    container.classList.add("is-hidden");
    // Disable apply button if no suggestions
    const applyBtn = document.getElementById("applySuggestionBtn");
    if (applyBtn) applyBtn.disabled = true;
    return;
  }
  
  console.log("Rendering suggestions:", suggestions);
  container.classList.remove("is-hidden");
  
  // Show latest iteration description
  const descContainer = document.getElementById("suggestionIterationDescription");
  if (descContainer) {
    const desc = suggestions.latest_iteration_description || "No description available";
    const iterNum = suggestions.iteration_number || "?";
    descContainer.innerHTML = `
      <div class="is-size-7">
        <strong>Iteration ${iterNum} description:</strong>
        <p class="mt-2" style="white-space: pre-wrap;">${escapeHtml(desc)}</p>
      </div>
    `;
  }
  
  // Show current prompts for context
  const currentPosEl = document.getElementById("suggestionCurrentPositive");
  const currentNegEl = document.getElementById("suggestionCurrentNegative");
  if (currentPosEl) {
    const currentPos = (suggestions.current_positive_terms || []).join(", ") || "None";
    currentPosEl.textContent = currentPos || "None";
  }
  if (currentNegEl) {
    const currentNeg = (suggestions.current_negative_terms || []).join(", ") || "None";
    currentNegEl.textContent = currentNeg || "None";
  }
  
  // Enable apply button when suggestions are rendered
  const applyBtn = document.getElementById("applySuggestionBtn");
  if (applyBtn) applyBtn.disabled = false;
  
  const posSuggestions = suggestions.positive_suggestions || {};
  const negSuggestions = suggestions.negative_suggestions || {};
  
  // Render positive suggestions
  const posContainer = document.getElementById("positiveSuggestions");
  if (posContainer) {
    renderSuggestionList(posContainer, posSuggestions, "positive", onApply);
  }
  
  // Render negative suggestions
  const negContainer = document.getElementById("negativeSuggestions");
  if (negContainer) {
    renderSuggestionList(negContainer, negSuggestions, "negative", onApply);
  }
}

// Render a single suggestion list
function renderSuggestionList(container, suggestions, type, onApply) {
  if (!container) return;
  
  const toAdd = suggestions.to_add || [];
  const toRemove = suggestions.to_remove || [];
  
  if (toAdd.length === 0 && toRemove.length === 0) {
    container.innerHTML = '<p class="has-text-grey is-size-7">No suggestions</p>';
    return;
  }
  
  let html = "";
  
  // Items to remove (marked with X)
  if (toRemove.length > 0) {
    html += '<div class="suggestions-remove mb-3">';
    html += '<p class="has-text-weight-bold is-size-7 mb-2">Suggested removals:</p>';
    html += '<div class="tags">';
    toRemove.forEach(term => {
      html += `
        <span class="tag is-danger is-light suggestion-tag" data-action="remove" data-term="${escapeHtml(term)}" data-type="${type}">
          ${escapeHtml(term)}
          <button class="delete is-small" data-action="remove" data-term="${escapeHtml(term)}" data-type="${type}"></button>
        </span>
      `;
    });
    html += '</div></div>';
  }
  
  // Items to add (grey, with checkbox)
  if (toAdd.length > 0) {
    html += '<div class="suggestions-add">';
    html += '<p class="has-text-weight-bold is-size-7 mb-2">Suggested additions:</p>';
    html += '<div class="tags">';
    toAdd.forEach(term => {
      html += `
        <span class="tag is-grey suggestion-tag" data-action="add" data-term="${escapeHtml(term)}" data-type="${type}">
          <label class="checkbox">
            <input type="checkbox" data-action="add" data-term="${escapeHtml(term)}" data-type="${type}">
          </label>
          <span class="ml-2">${escapeHtml(term)}</span>
        </span>
      `;
    });
    html += '</div></div>';
  }
  
  container.innerHTML = html;
  
  // Wire checkboxes and delete buttons
  container.querySelectorAll("[data-action]").forEach(el => {
    el.addEventListener("click", (e) => {
      const action = el.dataset.action;
      const term = el.dataset.term;
      const termType = el.dataset.type;
      
      if (action === "add" && el.tagName === "INPUT" && el.checked) {
        // Add term
        if (onApply) {
          onApply(termType, "add", term);
        }
      } else if (action === "remove") {
        // Remove term
        if (onApply) {
          onApply(termType, "remove", term);
        }
      }
    });
  });
}

// Generate suggestions
async function generateSuggestions(runId) {
  if (!runId) {
    return null;
  }
  
  const btn = document.getElementById("generateSuggestionBtn");
  const placeholder = document.getElementById("suggestionPlaceholder");
  
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = '<span class="icon"><i class="fas fa-spinner fa-spin"></i></span><span>Generating...</span>';
  }
  
  if (placeholder) {
    placeholder.textContent = "Generating AI suggestions...";
  }
  
  try {
    const project = window.__LIVE_PROJECT__;
    const res = await fetch(`/api/project/${project}/run/${runId}/suggest_prompts`);
    const data = await res.json();
    
    if (!res.ok) {
      if (placeholder) {
        placeholder.innerHTML = `<span class="has-text-danger">${escapeHtml(data.error || "Failed")}</span>`;
      }
      return null;
    }
    
    currentSuggestions = data;
    
    // Enable apply button when suggestions are ready
    const applyBtn = document.getElementById("applySuggestionBtn");
    if (applyBtn) applyBtn.disabled = false;
    
    return data;
  } catch (e) {
    if (placeholder) {
      placeholder.innerHTML = `<span class="has-text-danger">Error: ${escapeHtml(e.message)}</span>`;
    }
    return null;
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = '<span class="icon"><i class="fas fa-magic"></i></span><span>Generate</span>';
    }
  }
}

// Apply all checked suggestions
function applyAllSuggestions(onApply) {
  if (!currentSuggestions) return;
  
  const posSuggestions = currentSuggestions.positive_suggestions || {};
  const negSuggestions = currentSuggestions.negative_suggestions || {};
  
  // Get all checked items
  const checkedAdds = document.querySelectorAll("#positiveSuggestions input[type='checkbox']:checked, #negativeSuggestions input[type='checkbox']:checked");
  
  checkedAdds.forEach(checkbox => {
    const term = checkbox.dataset.term;
    const type = checkbox.dataset.type;
    if (onApply) {
      onApply(type, "add", term);
    }
  });
  
  // Apply all removals (they're marked with X)
  const removeButtons = document.querySelectorAll("#positiveSuggestions .suggestion-tag[data-action='remove'], #negativeSuggestions .suggestion-tag[data-action='remove']");
  removeButtons.forEach(btn => {
    const term = btn.dataset.term;
    const type = btn.dataset.type;
    if (onApply) {
      onApply(type, "remove", term);
    }
  });
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

// Escape HTML helper (if not already defined)
if (typeof escapeHtml === 'undefined') {
  function escapeHtml(text) {
    if (!text) return '';
    return String(text)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }
}

// Export
window.SuggestionsModule = {
  renderSuggestions,
  generateSuggestions,
  applyAllSuggestions,
  getCurrentSuggestions: () => currentSuggestions
};
