const facts = [
    "A standard soccer ball is made of 32 pentagonal and hexagonal panels.",
    "The 1970 Telstar was the first ball to use the iconic black and white pattern for TV visibility.",
    "The official weight of a soccer ball must be between 410g and 450g.",
    "Modern high-end balls are thermally bonded rather than stitched.",
    "The first rubber soccer ball was created by Charles Goodyear in 1855."
];

const btn = document.getElementById('factBtn');
const display = document.getElementById('factDisplay');

btn.addEventListener('click', () => {
    const randomIndex = Math.floor(Math.random() * facts.length);
    display.textContent = facts[randomIndex];
});
