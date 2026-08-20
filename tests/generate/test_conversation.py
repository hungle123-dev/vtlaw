from unittest.mock import MagicMock

from vtlaw.generate.conversation import ConversationRewriter


def test_rewrites_a_follow_up_with_history():
    llm = MagicMock()
    llm.complete.return_value = "mức phạt xe ô tô vượt đèn đỏ"

    result = ConversationRewriter(llm).rewrite(
        [{"role": "user", "content": "xe máy vượt đèn đỏ phạt bao nhiêu?"}],
        "còn xe ô tô thì sao?",
    )

    assert result == "mức phạt xe ô tô vượt đèn đỏ"


def test_history_is_quoted_as_data_not_replayed_as_chat_turns():
    """Replaying history as real turns makes the model answer instead of rewrite.

    Measured against gpt-4o-mini: with the turns replayed, the follow-up "Còn xe
    ô tô thì sao?" returned an invented penalty figure, which then became the
    retrieval query. Exactly two messages must reach the provider.
    """
    llm = MagicMock()
    llm.complete.return_value = "mức phạt xe ô tô vượt đèn đỏ"
    history = [
        {"role": "user", "content": "xe máy vượt đèn đỏ phạt bao nhiêu?"},
        {"role": "assistant", "content": "Từ 1.000.000 đến 2.000.000 đồng."},
    ]

    ConversationRewriter(llm).rewrite(history, "còn xe ô tô thì sao?")

    messages = llm.complete.call_args.kwargs["messages"]
    assert [m["role"] for m in messages] == ["system", "user"]
    prompt = messages[-1]["content"]
    assert "còn xe ô tô thì sao?" in prompt
    for turn in history:
        assert turn["content"] in prompt
    # A rewritten query is one line; an unbounded reply can be a paragraph of
    # invented law that goes straight into retrieval.
    assert llm.complete.call_args.kwargs["max_tokens"] <= 200


def test_keeps_original_question_without_history_or_on_failure():
    llm = MagicMock()
    rewriter = ConversationRewriter(llm)
    assert rewriter.rewrite([], "phạt bao nhiêu?") == "phạt bao nhiêu?"
    llm.complete.side_effect = RuntimeError("provider down")
    assert rewriter.rewrite([{"role": "user", "content": "xe máy"}], "còn ô tô?") == "còn ô tô?"
