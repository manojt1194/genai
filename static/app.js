const sessionId = crypto.randomUUID();

async function sendMessage() {
    const input = document.getElementById("message");
    const userMessage = input.value.trim();
    if (!userMessage) return;

    addMessage("You", userMessage, "user");
    input.value = "";

    const response = await fetch("/memory-chat", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            session_id: sessionId,
            message: userMessage
        })
    });

    const data = await response.json();
    addMessage("Bot", data.answer, "bot");
}

function addMessage(sender, text, cssClass) {
    const chatBox = document.getElementById("chat-box");
    const div = document.createElement("div");
    div.className = cssClass;
    div.innerHTML = `<strong>${sender}:</strong> ${text}`;
    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight;
}