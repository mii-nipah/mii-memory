# mii-memory

A smart memory management system for agents.
Made to replace the very own idea of context compression in agents.

## how it works?

Backed by a sqlite database, mii-memory works just like a "file system for memories" (with some twists)

You have generally 3 modes of storage:
* `global` - for storing memories that are very general in nature
* `workspace` - for memories that belong to a specific workspace (e.g. project, client, etc.)
* `session` - for memories that are specific to a session (e.g. a conversation with an agent)

Each memory entry can also have a specific expiration condition, which can be useful for memories that are only relevant while a certain condition holds true:
* `time`: expires after a certain amount of time has passed
* `usage`: expires after being accessed a certain number of times
* `file_exist`: lifetime tied to the existence of a specific file
* `file_pristine`: lifetime tied to the file being unchanged since the memory was created
* `period`: not necessarily an 'expiration', but the memory is only relevant during a specific time period, if you are in it exists, if you are not it does not exist (for the consumers)

Expiration conditions are checked upon retrieval, so it's not heavy on performance.

To insert a memory in the system, you have to provide:
* `content`: the actual memory content (string)
* `mode`: the storage mode
* `mode_ref`: the reference for the storage mode (e.g. workspace name or a session uuid)
* `tags`: the tags associated with the memory
* `expiration_condition`: the expiration condition (optional)
* `expiration_value`: the value for the expiration condition (optional)
* `metadata`: any additional metadata you want to store with the memory (optional)

## tags
Tags are a very important aspect of mii-memory. They work similarly to how directories work in a file system, for memories they allow models to chose relevant keywords that match the query and allow for an efficient retrieval of relevant memories when the time comes.

Agents are recommended to chose freely the best tags for their files, which makes navigation at query time much more efficient and pleasing.

## search
In query time, agents can then list available tags, filter them, filter by text directly or even both, and then receive what it's most relevant for them.
We also embed a very small CPU optimized version of the popular MiniLM model, and while storing data we create vector embeddings for the content and the tags, so at query time the results are not only filtered by precise matches, but also by semantic similarity.
It's also possible to use tags to negatively filter results, for example by including a tag in the negative filter, all memories with that tag will score lower in the results, thus being only found in the end of the results or not at all if limited.

## relevance
Unless an expiration is specified at insertion time, memories are never deleted, however they can become irrelevant, they can fade.
Memories have 2 intrinsic properties on them, positive and negative scores. When a memory is retrieved, it receives a positive score (according to the rank, the lower the gain in score).
During the set operation, if we find a memory with sufficiently high similarity, the other memory receives a negative score increase proportional to the likeliness that they share. This means that informations that are similar but not necessarily the same, or opposed, can coexist but they will cancel out each other's relevance over time.

During retrieval, both scores are taken into account, so a memory that has a high positive score but also a high negative score might not be as relevant as a memory with a lower positive score but also a much lower negative score.

## project

mii-memory is both a unix-like CLI tool and an MCP (Model Context Protocol), which means it can be used both as a standalone tool and also as a service that agents can interact with through the MCP protocol.

## commands
* `mii-memory set <content> [--mode <mode>] [<mode_ref>] [--tag <tag> ... at least 1 is expected] [--expiration-condition <expiration_condition> <expiration_value>] [--metadata <metadata>]`
* `mii-memory get <query> [--tag|--p-tag <tag> ...] [--n-tag <tag> ...] [--limit <limit>] [--offset <offset>]`
* `mii-memory list-tags [--filter <filter>]`

## mcp
mii-memory can also run as a service that agents can interact with through the MCP protocol.
The MCP commands are the same as the CLI commands, but they are sent as JSON payloads to the service endpoint.
Differently from the CLI, the MCP commands should not contain explicit references to the mode_ref, only the mode, since the service will be able to infer the mode_ref from the agent's identity and the current session. This reduces the error surface for agents and allows for a more seamless integration with the agent's workflow.
* `memory_set`
* `memory_get`
* `list_tags`
