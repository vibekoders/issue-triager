from triager_evals.normalize import body_commands, labels_from_commands, strip_slash_commands


def strip(body: str, mapping) -> str:
    return strip_slash_commands(body, mapping.slash_commands)


def test_removes_command_lines(mapping):
    body = "It broke.\n\n/kind bug\n/sig node\n  /priority important-soon\n/assign @bob\n/cc @carol\nThanks"
    assert strip(body, mapping) == "It broke.\n\nThanks"


def test_case_insensitive_and_all_commands(mapping):
    commands = [
        "/KIND bug",
        "/Sig api-machinery",
        "/area kubelet",
        "/triage accepted",
        "/label foo",
        "/remove-kind bug",
        "/remove-lifecycle stale",
        "/lifecycle frozen",
        "/milestone v1.33",
        "/close",
        "/reopen",
        "/retitle new title",
        "/help",
        "/good-first-issue",
        "/uncc",
        "/unassign",
        "/committee security-response",
        "/wg device-management",
        "/language ja",
    ]
    assert strip("\n".join(["keep me", *commands]), mapping) == "keep me"


def test_keeps_paths_and_inline_mentions(mapping):
    body = "/home/user/.kube/config is missing\nRun /kind in a shell? no\n/closed-loop control\nsee the /sig node docs"
    out = strip(body, mapping)
    assert "/home/user/.kube/config" in out
    assert "/closed-loop control" in out
    assert "see the /sig node docs" in out
    # "Run /kind" does not start the line, so it stays.
    assert "Run /kind in a shell? no" in out


def test_inside_code_blocks_and_comments(mapping):
    body = (
        "#### Relevant SIG(s)\n\n```\n/sig node\n```\n\n"
        "<!-- /kind bug -->\n"
        "<!--\n/sig storage\nUncomment the line above.\n-->\n"
        "> /priority backlog\n"
        "Text"
    )
    out = strip(body, mapping)
    assert "/sig" not in out and "/kind" not in out and "/priority" not in out
    assert "<!--\nUncomment the line above.\n-->" in out
    assert out.endswith("Text")


def test_crlf_and_blank_line_collapse(mapping):
    assert strip("a\r\n\r\n/kind bug\r\n\r\n\r\nb", mapping) == "a\n\nb"


def test_commands_to_labels(mapping):
    cmds = body_commands(
        "/kind bug\n/sig node storage\n<!-- /area kubelet -->\n/priority backlog", mapping.slash_commands
    )
    assert labels_from_commands(cmds) == {"kind/bug", "sig/node", "sig/storage", "area/kubelet", "priority/backlog"}


def test_misspelled_label_commands_are_removed_but_add_no_labels(mapping):
    body = "Text\n/sig-network\n/sig/testing\n/triage-accepted\n/kind-bug\n/close-loop stays"
    assert strip(body, mapping) == "Text\n/close-loop stays"
    assert labels_from_commands(body_commands(body, mapping.slash_commands)) == set()


def test_committee_and_wg_commands_to_labels(mapping):
    cmds = body_commands("/committee security-response\n/wg structured-logging", mapping.slash_commands)
    assert labels_from_commands(cmds) == {"committee/security-response", "wg/structured-logging"}
    assert strip("Text\n/committee-security-response", mapping) == "Text"
