/* SPDX-License-Identifier: AGPL-3.0-only
 * Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
 *
 * Minimal C client for SAPIENT. Streams a reply from a local model.
 *
 *   make && ./c-chat "Why is the sky blue?"
 *
 * This file is also a canary: if it stops compiling or linking, the C ABI broke.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sapient.h"

/* Print the error detail (if any), free it, and return a process exit code. */
static int fail(const char *what, sapient_error_t *err) {
    const char *msg = err ? sapient_error_message(err) : NULL;
    fprintf(stderr, "%s failed (%d): %s\n",
            what,
            err ? sapient_error_code(err) : SAPIENT_ERR_INTERNAL,
            msg ? msg : "no detail");
    sapient_error_free(err);
    return 1;
}

/* Called for each decoded fragment. `token` is only valid for this call, so we print it
 * immediately rather than storing the pointer. Returning false would cancel generation. */
static bool on_token(const char *token, void *user_data) {
    (void)user_data;
    fputs(token, stdout);
    fflush(stdout);
    return true;
}

int main(int argc, char **argv) {
    const char *model  = getenv("SAPIENT_MODEL");
    const char *prompt = argc > 1 ? argv[1] : "Say hello in one short sentence.";
    if (!model) model = "smollm2-135m-q4";  /* small: quick to download for a demo */

    printf("sapient %s (ABI %u)\n", sapient_version(), sapient_api_version());
    if (sapient_api_version() != SAPIENT_API_VERSION) {
        fprintf(stderr,
                "ABI mismatch: header is %u, library is %u — rebuild against the "
                "matching sapient.h\n",
                SAPIENT_API_VERSION, sapient_api_version());
        return 1;
    }

    sapient_error_t *err = NULL;

    /* Always start from the defaults so this stays source-compatible as fields are
     * added, then override only what we care about. */
    sapient_options_t opts = sapient_options_default();
    opts.max_tokens  = 128;
    opts.temperature = 0.7f;   /* any sampling field set => sampled, not greedy */
    opts.backend     = SAPIENT_BACKEND_AUTO;

    printf("loading %s (first run downloads weights)...\n", model);
    sapient_session_t *s = sapient_session_load(model, &opts, &err);
    if (!s) return fail("load", err);

    char *backend = sapient_session_backend(s);
    printf("backend: %s\n\n", backend ? backend : "unknown");
    sapient_string_free(backend);

    printf("> %s\n", prompt);

    char *reply = NULL;
    if (sapient_chat_stream(s, prompt, on_token, NULL, &reply, &err) != SAPIENT_OK) {
        sapient_session_free(s);
        return fail("chat", err);
    }
    printf("\n\n(%zu bytes; transcript has %zu messages)\n",
           strlen(reply), sapient_transcript_len(s));

    /* Everything handed to us is ours to free, exactly once. */
    sapient_string_free(reply);
    sapient_session_free(s);
    return 0;
}
