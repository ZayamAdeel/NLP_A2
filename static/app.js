const chatWindow = document.getElementById("chat-window");
const input = document.getElementById("msg-input");
const sendBtn = document.getElementById("send-btn");
const newChatBtn = document.getElementById("new-chat-btn");
const stageBadge = document.getElementById("stage-badge");
const connStatus = document.getElementById("conn-status");

let sessionId = null;
let socket = null;
let currentAssistantBubble = null;

function addBubble(role, text) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = text;
  chatWindow.appendChild(div);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return div;
}

function addMeta(text) {
  const div = document.createElement("div");
  div.className = "meta";
  div.textContent = text;
  chatWindow.appendChild(div);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function setSending(isSending) {
  sendBtn.disabled = isSending;
  input.disabled = isSending;
}

async function bootstrapSession() {
  chatWindow.innerHTML = "";
  const res = await fetch("/api/session", { method: "POST" });
  const data = await res.json();
  sessionId = data.session_id;
  addBubble("assistant", data.greeting);
  connectSocket();
}

function connectSocket() {
  if (socket) {
    socket.close();
  }
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${protocol}://${window.location.host}/ws/chat`);

  socket.onopen = () => {
    connStatus.textContent = "connected";
  };

  socket.onclose = () => {
    connStatus.textContent = "disconnected — retrying in 2s";
    setTimeout(connectSocket, 2000);
  };

  socket.onerror = () => {
    connStatus.textContent = "connection error";
  };

  socket.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    handleServerEvent(msg);
  };
}

function handleServerEvent(msg) {
  switch (msg.type) {
    case "stage":
      stageBadge.textContent = `stage: ${msg.stage}`;
      break;
    case "token":
      if (!currentAssistantBubble) {
        currentAssistantBubble = addBubble("assistant", "");
      }
      currentAssistantBubble.textContent += msg.text;
      chatWindow.scrollTop = chatWindow.scrollHeight;
      break;
    case "blocked":
      addBubble("blocked", msg.text);
      setSending(false);
      currentAssistantBubble = null;
      break;
    case "done": {
      const ttft = msg.time_to_first_token != null ? `${msg.time_to_first_token}s TTFT` : "";
      const tps = msg.tokens_per_second != null ? `${msg.tokens_per_second} tok/s` : "";
      const total = msg.total_time != null ? `${msg.total_time}s total` : "";
      addMeta([ttft, tps, total].filter(Boolean).join(" · "));
      currentAssistantBubble = null;
      setSending(false);
      break;
    }
    case "error":
      addBubble("error", `⚠ ${msg.message}`);
      currentAssistantBubble = null;
      setSending(false);
      break;
    default:
      console.warn("Unknown event", msg);
  }
}

function sendMessage() {
  const text = input.value.trim();
  if (!text || !sessionId || socket.readyState !== WebSocket.OPEN) return;

  addBubble("user", text);
  input.value = "";
  setSending(true);
  currentAssistantBubble = null;

  socket.send(JSON.stringify({ type: "message", session_id: sessionId, text }));
}

sendBtn.addEventListener("click", sendMessage);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendMessage();
});

newChatBtn.addEventListener("click", async () => {
  if (sessionId) {
    await fetch(`/api/session/${sessionId}`, { method: "DELETE" });
  }
  stageBadge.textContent = "stage: greeting";
  await bootstrapSession();
});

bootstrapSession();
