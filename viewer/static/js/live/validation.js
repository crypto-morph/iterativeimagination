/* global window */
// Module for real-time validation of prompt terms

// Validation rules
const ValidationRules = {
  // Check if a term appears in both positive and negative
  checkContradiction(positiveTerms, negativeTerms) {
    const issues = [];
    const posLower = new Set(positiveTerms.map(t => t.toLowerCase()));
    const negLower = new Set(negativeTerms.map(t => t.toLowerCase()));
    
    for (const term of positiveTerms) {
      if (negLower.has(term.toLowerCase())) {
        issues.push({
          type: "contradiction",
          term: term,
          message: `"${term}" appears in both positive and negative prompts`
        });
      }
    }
    
    return issues;
  },
  
  // Check if must_include terms are in negative
  checkMustIncludeInNegative(mustIncludeTerms, negativeTerms) {
    const issues = [];
    const negLower = new Set(negativeTerms.map(t => t.toLowerCase()));
    
    for (const term of mustIncludeTerms) {
      if (negLower.has(term.toLowerCase())) {
        issues.push({
          type: "must_include_conflict",
          term: term,
          message: `Must-include term "${term}" is in negative prompt`
        });
      }
    }
    
    return issues;
  },
  
  // Check if ban_terms are in positive
  checkBanTermsInPositive(banTerms, positiveTerms) {
    const issues = [];
    const posLower = new Set(positiveTerms.map(t => t.toLowerCase()));
    
    for (const term of banTerms) {
      if (posLower.has(term.toLowerCase())) {
        issues.push({
          type: "ban_term_conflict",
          term: term,
          message: `Ban term "${term}" is in positive prompt`
        });
      }
    }
    
    return issues;
  },
  
  // Run all validations
  validateAll(positiveTerms, negativeTerms, maskTerms) {
    const allIssues = [];
    
    // Check contradictions
    allIssues.push(...this.checkContradiction(positiveTerms, negativeTerms));
    
    // Check must_include conflicts
    if (maskTerms && maskTerms.must_include) {
      allIssues.push(...this.checkMustIncludeInNegative(maskTerms.must_include, negativeTerms));
    }
    
    // Check ban_terms conflicts
    if (maskTerms && maskTerms.ban_terms) {
      allIssues.push(...this.checkBanTermsInPositive(maskTerms.ban_terms, positiveTerms));
    }
    
    return allIssues;
  }
};

// Display validation issues
function displayValidationIssues(container, issues) {
  if (!container) return;
  
  if (issues.length === 0) {
    container.innerHTML = '<div class="has-text-success is-size-7"><i class="fas fa-check-circle"></i> No validation issues</div>';
    container.classList.remove("has-background-danger-light");
    container.classList.add("has-background-success-light");
    return;
  }
  
  container.innerHTML = `
    <div class="has-text-danger is-size-7">
      <i class="fas fa-exclamation-triangle"></i> <strong>Validation Issues:</strong>
      <ul style="margin-top: 0.5rem;">
        ${issues.map(issue => `<li>${escapeHtml(issue.message)}</li>`).join("")}
      </ul>
    </div>
  `;
  container.classList.remove("has-background-success-light");
  container.classList.add("has-background-danger-light");
}

// Highlight problematic terms in lists
function highlightProblematicTerms(issues, termType) {
  // This will be called to add visual indicators to terms
  // For now, we'll just return the issues
  return issues.filter(issue => {
    if (termType === "positive") {
      return issue.type === "contradiction" || issue.type === "ban_term_conflict";
    } else if (termType === "negative") {
      return issue.type === "contradiction" || issue.type === "must_include_conflict";
    }
    return false;
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

// Export
window.ValidationModule = {
  ValidationRules,
  displayValidationIssues,
  highlightProblematicTerms
};
