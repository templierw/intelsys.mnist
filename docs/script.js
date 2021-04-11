const CANVAS_SIZE = 280;
const CANVAS_SCALE = 0.5;

const modelSelector = document.getElementById('model-select');

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const clearButton = document.getElementById("clear-button");

let isMouseDown = false;
let hasIntroText = true;
let lastX = 0;
let lastY = 0;

// Load our model.
let modelsToLoad = 0;
const sessions = [];
const loadingPromises = [];
let selectedSession = 0;
let processedInput = false;

for (let i = 0; i < modelSelector.options.length; i++) {
  let sess = new onnx.InferenceSession();
  sessions.push(sess);
  loadingPromises.push(sess.loadModel('./onnx/'+modelSelector.options[i].innerText+'.onnx'));
  modelsToLoad++;
}
modelSelector.addEventListener('change', (event) => {
  selectedSession = Number.parseInt(event.target.value);
  console.log("changed?");
  if (processedInput) {
    console.log("changed2?");
    updatePredictions();
  }
});

// Add 'Draw a number here!' to the canvas.
ctx.lineWidth = 28;
ctx.lineJoin = "round";
ctx.font = "28px sans-serif";
ctx.textAlign = "center";
ctx.textBaseline = "middle";
ctx.fillStyle = "#000000";
ctx.fillText("Loading...", CANVAS_SIZE / 2, CANVAS_SIZE / 2);

// Set the line color for the canvas.
ctx.strokeStyle = "#000000";

function clearCanvas() {
  ctx.fillStyle = "#FFFFFF";
  ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  for (let i = 0; i < 10; i++) {
    const element = document.getElementById(`prediction-${i}`);
    element.className = "prediction-col";
    element.children[0].children[0].style.height = "0";
  }
  processedInput = false;
}

function drawLine(fromX, fromY, toX, toY) {
  // Draws a line from (fromX, fromY) to (toX, toY).
  ctx.beginPath();
  ctx.moveTo(fromX, fromY);
  ctx.lineTo(toX, toY);
  ctx.closePath();
  ctx.stroke();
  updatePredictions();
}

function imageDataParse(imageData) {
  let data = []; while(data.push([]) < CANVAS_SIZE);
  for (let i = 0; i < CANVAS_SIZE*CANVAS_SIZE; i++) {
    data[i % CANVAS_SIZE][Math.floor(i/CANVAS_SIZE)] = [imageData[i*4],
      imageData[i*4+1],
      imageData[i*4+2],
      imageData[i*4+3]];
  }
  return data;
}

const SCALE_FACTOR = CANVAS_SIZE/28;

function averageData(data) {
  data = imageDataParse(data.data);
  const rval = []
  for (let col = 0; col < 28; col++) {
    const rowArr = [];
    for (let row = 0; row < 28; row++) {
      let pixVal = 0;
      for (let i = 0; i < SCALE_FACTOR*SCALE_FACTOR; i++) {
          pixVal += (255-data[row*SCALE_FACTOR + Math.floor(i/SCALE_FACTOR)][(col*SCALE_FACTOR + i%SCALE_FACTOR)][0]);
          //if (row == 0 && col == 0)
            //console.log("Considering " + ((row*10 + Math.floor(i/10)) * 28 + (col*10 + i%10)) + ": " + data.data[((row*10 + Math.floor(i/10)) * 28 + (col*10 + i%10))*4]);
      }
      rowArr.push((((pixVal/100/256.0)) - 0.13063)/0.30811);
    }
    rval.push(rowArr)
  }
  return rval;
}

async function updatePredictions() {
  // Get the predictions for the canvas data.
  const imgData = ctx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  const averagedData = averageData(imgData);

  //console.log(averagedData);

  const input = new onnx.Tensor(new Float32Array(averagedData.flat()), "float32", [1,1,28,28]);

  const outputMap = await sessions[selectedSession].run([input]);
  processedInput = true;
  const outputTensorIterator = outputMap.values();
  outputTensorIterator.next();
  const outputTensor = outputTensorIterator.next().value;
  const predictions = outputTensor.data;
  const maxPrediction = Math.max(...predictions);

  for (let i = 0; i < predictions.length; i++) {
    const element = document.getElementById(`prediction-${i}`);
    element.children[0].children[0].style.height = `${predictions[i] * 100}%`;
    element.className =
      predictions[i] === maxPrediction
        ? "prediction-col top-prediction"
        : "prediction-col";
  }
}

function canvasMouseDown(event) {
  event.preventDefault();
  event.stopPropagation();
  isMouseDown = true;
  if (hasIntroText) {
    clearCanvas();
    hasIntroText = false;
  }
  const x = event.offsetX / CANVAS_SCALE;
  const y = event.offsetY / CANVAS_SCALE;

  // To draw a dot on the mouse down event, we set laxtX and lastY to be
  // slightly offset from x and y, and then we call `canvasMouseMove(event)`,
  // which draws a line from (laxtX, lastY) to (x, y) that shows up as a
  // dot because the difference between those points is so small. However,
  // if the points were the same, nothing would be drawn, which is why the
  // 0.001 offset is added.
  lastX = x + 0.001;
  lastY = y + 0.001;
  canvasMouseMove(event);
}

function canvasMouseMove(event) {
  event.preventDefault();
  event.stopPropagation();
  const x = event.offsetX / CANVAS_SCALE;
  const y = event.offsetY / CANVAS_SCALE;
  if (isMouseDown) {
    drawLine(lastX, lastY, x, y);
  }
  lastX = x;
  lastY = y;
}

function bodyMouseUp() {
  isMouseDown = false;
}

function bodyMouseOut(event) {
  // We won't be able to detect a MouseUp event if the mouse has moved
  // ouside the window, so when the mouse leaves the window, we set
  // `isMouseDown` to false automatically. This prevents lines from
  // continuing to be drawn when the mouse returns to the canvas after
  // having been released outside the window.
  if (!event.relatedTarget || event.relatedTarget.nodeName === "HTML") {
    isMouseDown = false;
  }
}

const dummyInputArr = [];
for (let i = 0; i < 28*28; i++)
  dummyInputArr.push(0);
const dummyInput = new onnx.Tensor(new Float32Array(dummyInputArr), "float32", [1,1,28,28]);

for (let i = 0; i < modelsToLoad; i++) {

  const loadCallback = function () {
    sessions[this.sessionId].run([dummyInput]).then(() => {
      modelsToLoad--;
      if (modelsToLoad == 0) {
        canvas.addEventListener("mousedown", canvasMouseDown);
        canvas.addEventListener('touchstart', canvasMouseDown);
        canvas.addEventListener('touchmove', function (e) {
          e.preventDefault();
          e.stopPropagation();
          let touch = e.touches[0];
          let mouseEvent = new MouseEvent("mousemove", {
            clientX: touch.clientX,
            clientY: touch.clientY
          });
          canvas.dispatchEvent(mouseEvent);
        }, false);
        canvas.addEventListener("mousemove", canvasMouseMove);
        document.body.addEventListener("mouseup", bodyMouseUp);
        document.body.addEventListener('mouseout', bodyMouseUp);
        document.body.addEventListener("mouseout", bodyMouseOut);
        clearButton.addEventListener("mousedown", clearCanvas);
      
        ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
        ctx.fillText("Draw a number!", CANVAS_SIZE / 2, CANVAS_SIZE / 2);
      }
    });
  }

  loadingPromises[i].then(loadCallback.bind({sessionId: i}));
}

