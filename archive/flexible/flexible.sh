#!/usr/bin/env bash
set -euo pipefail

IMG_PATH="${1:-example.jpg}"
MODEL="${2:-qwen3-vl:4b}"
RULES_FILE="${3:-rules.yaml}"

OUTDIR="${OUTDIR:-out}"
mkdir -p "$OUTDIR"
TS=$(date -u +%Y%m%dT%H%M%SZ)
BASE=$(basename "$IMG_PATH")
PREFIX="$OUTDIR/$BASE.$TS"

if [[ ! -f "$IMG_PATH" ]]; then
  echo "ERROR: Image file not found: $IMG_PATH" >&2
  exit 1
fi

if [[ ! -f "$RULES_FILE" ]]; then
  echo "ERROR: Rules file not found: $RULES_FILE" >&2
  exit 1
fi

# Generate prompt and schema from rules
PROMPT=$(python3 -c "
import yaml, json

with open('$RULES_FILE') as f:
    rules = yaml.safe_load(f)

def build_schema(questions):
    schema = {}
    for q in questions:
        if q['type'] == 'array' and 'items' in q:
            schema[q['field']] = {'type': 'array', 'items': build_schema(q['items'])}
        else:
            schema[q['field']] = {'type': q['type']}
            if 'enum' in q:
                schema[q['field']]['enum'] = q['enum']
    return schema

def build_prompt(questions, indent=0):
    lines = []
    for q in questions:
        prefix = '  ' * indent + '- '
        if q['type'] == 'array' and 'items' in q:
            lines.append(f'{prefix}{q[\"field\"]} (array): {q[\"question\"]}')
            for item in q['items']:
                lines.append(f'  {prefix}{item[\"field\"]} ({item[\"type\"]}): {item[\"question\"]}')
                if 'enum' in item:
                    lines.append(f'    {prefix}Options: {item[\"enum\"]}')
        else:
            lines.append(f'{prefix}{q[\"field\"]} ({q[\"type\"]}): {q[\"question\"]}')
            if 'enum' in q:
                lines.append(f'  {prefix}Options: {q[\"enum\"]}')
    return '\\n'.join(lines)

schema = build_schema(rules['questions'])
prompt_text = build_prompt(rules['questions'])

print(f'''Analyze the image and return ONLY valid JSON matching this structure:

{prompt_text}

Return STRICT JSON only—no comments, no markdown, no extra text.''')
")

echo "STARTING"

if base64 --help 2>&1 | grep -q ' -w, --wrap'; then
  IMG_B64=$(base64 -w 0 "$IMG_PATH")
else
  IMG_B64=$(base64 < "$IMG_PATH" | tr -d '\n')
fi

REQUEST_BODY=$(
  printf '%s' "$IMG_B64" |
  jq -Rn --arg model "$MODEL" --arg prompt "$PROMPT" '
    {
      model: $model,
      prompt: $prompt,
      stream: false,
      images: [ input ]
    }
  '
)

printf '%s' "$REQUEST_BODY" > "${PREFIX}.request.json"

RESPONSE=$(
  printf '%s' "$REQUEST_BODY" |
  curl -sS -w '\n%{http_code}' -H "Content-Type: application/json" \
       -X POST --data-binary @- http://localhost:11434/api/generate
)

HTTP_CODE=$(printf '%s\n' "$RESPONSE" | tail -n1)
BODY=$(printf '%s\n' "$RESPONSE" | sed '$d')

printf '%s' "$BODY" > "${PREFIX}.body.json"

if [[ "$HTTP_CODE" != "200" ]]; then
  echo "ERROR: HTTP $HTTP_CODE from Ollama." >&2
  exit 1
fi

RAW_TEXT=$(jq -r '.response // empty' "${PREFIX}.body.json")

if [[ -z "$RAW_TEXT" ]]; then
  echo "No response from model" >&2
  exit 1
fi

if echo "$RAW_TEXT" | jq empty >/dev/null 2>&1; then
  echo "$RAW_TEXT" | jq . | tee "${PREFIX}.parsed.json"
else
  echo "$RAW_TEXT"
  echo "NOTE: Non-JSON response saved at: ${PREFIX}.response.txt" >&2
  printf '%s' "$RAW_TEXT" > "${PREFIX}.response.txt"
fi

echo "Artifacts: ${PREFIX}.*"
