Act as an expert system architect. Initialize my "Personal OS & Second Brain" in this current empty directory.

Execute the following steps exactly as described. Do not ask for confirmation between steps; only create the requested folders and files.

### STEP 1: Create the PARA directory structure

Create these folders:

- `00_Inbox`
- `01_Projects/ai-engineering-course`
- `02_Areas/Career`
- `03_Resources/RAG_and_Agents`
- `04_Archive`
- `99_System/Templates`
- `.cursor/rules`

### STEP 2: Create agent instructions

Create `AGENTS.md` and `CLAUDE.md` in the root directory with identical content:

```markdown
# Second Brain Context

You are my Second Brain assistant. This workspace uses the PARA method.

## Rules

1. Store all notes as Markdown files.
2. Add YAML frontmatter to every new note with `tags: []` and `date: YYYY-MM-DD`.
3. Use Obsidian-style links such as `[[File Name]]` to connect related notes.
4. When answering questions about my projects, learning, decisions, or previous research:
   - Search the Second Brain before relying on general knowledge.
   - If Basic Memory tools are available, prefer semantic search and context retrieval through Basic Memory.
   - Otherwise search Markdown files in `01_Projects`, `02_Areas`, and `03_Resources`.
   - Clearly distinguish information retrieved from my Second Brain from your own general knowledge.
5. Keep answers concise and cite the original notes with `[[wiki links]]`.
6. Never delete or overwrite an existing note without asking for confirmation.
7. Never store secrets, credentials, access tokens, or confidential employer/client information. Ask for confirmation before storing other sensitive personal data and keep only the minimum necessary details.
```

Create `.cursor/rules/second-brain.mdc` with this content:

```markdown
---
description: Second Brain retrieval and note-taking rules
alwaysApply: true
---

This workspace is a personal Second Brain organized with the PARA method.

- Store notes as Markdown with YAML frontmatter containing `tags` and `date`.
- Use `[[wiki links]]` to connect related notes.
- For questions about projects, learning, decisions, or previous research, search the Second Brain first.
- Prefer Basic Memory retrieval tools when available; otherwise search `01_Projects`, `02_Areas`, and `03_Resources`.
- Distinguish retrieved knowledge from general knowledge and cite the original notes.
- Never delete or overwrite an existing note without confirmation.
- Never store secrets, credentials, access tokens, or confidential employer/client information. Confirm before storing other sensitive personal data.
```

### STEP 3: Create the workspace README

Create `README.md` in the root directory:

````markdown
# Personal Second Brain

This is my private, local-first knowledge base. Markdown files are the source of truth, so the notes remain readable and portable without any AI tool.

## Workflow

```text
Capture → Process → Connect → Retrieve → Review
```

1. **Capture:** save raw ideas and materials in `00_Inbox/`.
2. **Process:** convert inbox items into structured notes under `01_Projects/`, `02_Areas/`, or `03_Resources/`.
3. **Connect:** add tags and `[[wiki links]]` between related notes.
4. **Retrieve:** search the Markdown files directly or use Basic Memory when available.
5. **Review:** regularly process the inbox, update active projects, and archive inactive material.

## Structure

- `00_Inbox/` — unprocessed notes and ideas
- `01_Projects/` — active work with a defined outcome
- `02_Areas/` — ongoing responsibilities and interests
- `03_Resources/` — reusable knowledge and references
- `04_Archive/` — inactive or processed material
- `99_System/` — templates and agent routines

## Working with AI

Open this root directory as the AI agent's workspace. The agent instructions in `AGENTS.md`, `CLAUDE.md`, and `.cursor/rules/second-brain.mdc` tell supported agents to search this knowledge base before answering questions about my projects, learning, decisions, or previous research.

Basic Memory is optional. When configured, it adds semantic retrieval and relationships while Markdown remains the source of truth.

## Privacy

- Keep this workspace separate from homework repositories.
- Do not publish it or push it to a public repository.
- Never store passwords, API keys, credentials, access tokens, or private keys.
- Do not store confidential employer, client, or customer information.
- Minimize sensitive personal data and review notes before sharing or backing them up remotely.
````

### STEP 4: Create core templates

Create `99_System/Templates/Project_Template.md`:

```markdown
---
status: active
tags: [project]
date: YYYY-MM-DD
---

# 🚀 [Project Name]

**Goal:** What is the definition of done?

## 📋 Action Items

- [ ] Task 1

## 🔗 Resources

- [[Resource 1]]
```

Create `99_System/Templates/Concept_Node.md`:

```markdown
---
tags: [resource, concept]
date: YYYY-MM-DD
---

# 💡 [Concept Name]

## 📝 Summary

(3 sentences max)

## 🧠 Implementation / Code

(How to use this in practice)
```

### STEP 5: Create the agents and prompts library

Create `99_System/agents.md`:

```markdown
# 🤖 AI Agent Prompts & Routines

Copy one of these prompts into an AI chat to run a routine.

## 🧹 1. Inbox Processor

Read all Markdown files in `00_Inbox/`. For each file:

1. Extract the core concepts and assign relevant tags.
2. Create a structured note using `99_System/Templates/Concept_Node.md`.
3. Save the new note in the most relevant folder under `02_Areas/` or `03_Resources/`.
4. Add useful `[[wiki links]]` to related notes.
5. After verifying that the new note was created successfully, move the original file to a dated folder under `04_Archive/Inbox/`. Never delete the original.

## 🔍 2. Knowledge Retrieval

Search `01_Projects`, `02_Areas`, and `03_Resources` for information about [INSERT TOPIC].

If Second Brain retrieval tools are available, start with semantic search and follow relevant relationships between notes. Otherwise search the Markdown files directly.

Synthesize the findings into a concise answer, cite original notes using `[[wiki links]]`, and clearly separate retrieved knowledge from your own general knowledge.

## 🏗️ 3. Project Scaffolding

I want to start a new project about [INSERT PROJECT IDEA].

Create a new folder in `01_Projects/`, initialize its `README.md` using `99_System/Templates/Project_Template.md`, and suggest three initial action items based on related knowledge in `02_Areas/` and `03_Resources/`.
```

### STEP 6: Verify the setup

Verify that every requested folder and file exists. Then return:

1. A compact tree of the generated structure.
2. A list of the instruction files created for Codex, Claude Code, and Cursor.
3. Confirmation that no files outside the current directory were modified.
