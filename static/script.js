// DOM Elements

const body = document.querySelector("body");
const container = document.querySelector("div.canvas");
const outputDiv = document.querySelector("div.prediction");

const canvas = document.querySelector("canvas");
const ctx = canvas.getContext('2d');

// Application state

let points = [];
let drawing = false;

// Event listeners

canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mousemove', addPoint);
window.addEventListener('resize', resizeCanvas);

// Initial Setup

resizeCanvas();
clearCanvas();

// Functions

// This cannot be done in CSS, because that would mess up
// the coordinate system on the canvas. 
function resizeCanvas() {
    function outerHeight(el) {
        let height = el.offsetHeight;
        var style = getComputedStyle(el);
        
        height += parseInt(style.marginTop) + parseInt(style.marginBottom);
        return height;
    }

    let height = container.clientHeight - 6;
    for (const child of document.querySelectorAll("div.canvas > :not(:first-child)"))
    {
        height -= outerHeight(child);
    }

    const size = Math.min(container.clientWidth, height);

    canvas.width = size;
    canvas.height = size;
    canvas.style.width = size;
    canvas.style.height = size;

    const button = document.querySelector("button.clear");
    button.style.width = size + 6;

    clearCanvas();
}

function startDrawing() {
    points = [];
    drawing = true;
}

function stopDrawing() {
    drawing = false;
    predict();
}

// draw a line between the last 10 points
function drawPoints() {
    const tail = points.slice(-10);

    ctx.lineWidth = 20;
    ctx.beginPath();

    const [startX, startY] = tail[0];
    ctx.moveTo(startX, startY);
    for (const [x, y] of tail.slice(1)) {
        ctx.lineTo(x, y);
    }

    ctx.stroke();
}

function addPoint(event) {
    if (drawing) {
        const x = event.clientX - canvas.offsetLeft;
        const y = event.clientY - canvas.offsetTop;

        points.push([x, y]);
        drawPoints();
    }
}

function clearCanvas() {
    outputDiv.innerHTML = "";
    ctx.fillStyle = "white"
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

async function predict() {
    const digit = canvas.toDataURL('image/png');
    const rawResponse = await fetch('/predict', {
        method: 'POST',
        headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
        },
        body: JSON.stringify({ digit })
    });
    const content = await rawResponse.json();
    console.log(content);

    const outputDiv = document.querySelector("div.prediction");
    outputDiv.innerHTML = content.prediction;
}
