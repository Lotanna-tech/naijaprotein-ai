
document.addEventListener("DOMContentLoaded", () => {
    const messages = document.querySelectorAll(".flash-message");

    messages.forEach((message) => {
        setTimeout(() => {
            message.style.transition = "opacity 0.4s ease";
            message.style.opacity = "0";

            setTimeout(() => message.remove(), 400);
        }, 5000);
    });
});
