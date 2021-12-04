const body = document.querySelector("body");
const canvasContainer = document.querySelector("div.canvas");
const canvas = document.querySelector("canvas");

// Resize canvas.
// This cannot be done in CSS, because that would mess up
// the coordinate system on the canvas. 
const size = Math.min(canvasContainer.clientWidth, canvasContainer.clientHeight);
canvas.width = size;
canvas.height = size;

const ctx = canvas.getContext('2d');
ctx.lineWidth = 20;
clearCanvas();

let points = [];
let drawing = false;

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
    ctx.fillStyle = "white"
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

async function predict() {
    const digit = canvas.toDataURL('image/png');
    const rawResponse = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
        },
        body: JSON.stringify({ digit })
    });
    const content = await rawResponse.json();
    console.log(content);
}

canvas.onmousedown = startDrawing;
canvas.onmouseup = stopDrawing;
canvas.onmousemove = addPoint;
