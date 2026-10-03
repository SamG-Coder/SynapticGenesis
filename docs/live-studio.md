# Local Learning Studio

A local HTML interface for native C++/CUDA questions and explicit corrections. The page is in `ui/index.html`. Browser JavaScript renders the interface; all model inference and weight updates run in the native executable. Python serves loopback HTTP and orchestrates files/processes.

Build with `./build.ps1 -ResidentConversation -SkipTests`, then start:

```powershell
python scripts/live_studio.py --workspace D:/SynapticGenesis
```

If the runtime is in a different checkout, add `--runtime PATH/TO/synaptic-resident-conversation.exe`. Open http://127.0.0.1:8765. The server binds only to loopback. The repository includes the HTML, CSS, JavaScript, server and model catalog; multi-gigabyte checkpoints and training books are local assets, not included in GitHub. A static HTML preview cannot execute native CUDA.

Select a catalog model and Load model. Ask questions, choose Correct this answer, enter your intended answer and click Teach correction. Practice counts range from 1 to 256. Corrections are explicitly admitted user feedback, with source bytes and hashes recorded before training. Questions alone do not train. Generated text never becomes a target automatically.

The model remains resident between questions. To admit a new correction, this version saves the resident state, appends a curriculum stage, reloads the same state with that stage and performs native updates. This reload is not continuous in-place admission. The UI reports the saving and learning phases. Each question resets its recurrent context; previous chat messages are displayed but not included in later prompts. Learned weights persist. Original checkpoints are never overwritten.

Each correction is saved before its automatic retest. Use Save checkpoint to take another snapshot. Saved sessions can be resumed from the sidebar. Files live in `runs/live-studio/<session-id>`. A failed operation preserves the last recorded checkpoint and pending feedback. Closing the browser leaves the local server and loaded model running. Stop the server with Ctrl+C for a graceful save; forcibly killing the process can lose unsaved work.

The model catalog pins checkpoint and curriculum hashes. Only completed, selected prose checkpoints and the explicitly corrected descendant are supplied. To add another model, review its lineage and register its exact hashes; retired source descendants must not be included. Current support requires grouped replay checkpoints, 128-byte training chunks, no teachers/SI, and the original `The bird ` speech prompt. Models with incomplete curricula cannot append corrections. Duplicate exact lessons are rejected by the curriculum builder; use more practice on first submission or a distinct lesson.

Reported speed is generated **bytes per second**, including native decode sampling/synchronization, excluding loading, graph capture and prefill. Full output is shown, including repetition. Raw hex and UTF-8 validity are available under each answer. No quality or generalization guarantee follows from a successful update.
