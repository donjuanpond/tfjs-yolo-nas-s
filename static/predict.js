import { renderBoxes } from "./renderBox.js";


$("#image-selector").change(function () {
	let reader = new FileReader();
	reader.onload = function () {
		let dataURL = reader.result;
		const back_img = new Image();
		back_img.src = dataURL
		back_img.onload = function () {
			let c = document.createElement('canvas')
			c.height = 640, c.width = 640
			const ctx = c.getContext('2d')

			var dx=0, dy=0, dWidth=0, dHeight=0;
			if (this.height >= this.width) {     // if its a tall image
				dWidth = (640 * this.width) / this.height
				dx = Math.floor((640-dWidth)/2)
				dy = 0
				dHeight = 640
			} else {									// if its a wide image
				dHeight = (640 * this.height) / this.width
				dx = 0
				dy = Math.floor((640-dHeight)/2)
				dWidth = 640
			}
			ctx.drawImage(this, dx, dy, dWidth, dHeight)
			
			$("#selectedImage").attr("src", c.toDataURL());
		}
		
	}
	let file = $("#image-selector").prop('files')[0];
	reader.readAsDataURL(file);
});


$("#webcam-capture-button").click(async function () {
	const webcamElement = document.getElementById("webcam")
	const webcam = await tf.data.webcam(webcamElement) 
	
	let c = document.createElement('canvas')
	const v = document.querySelector('video')
	c.height = v.videoHeight || parseInt(v.style.height)
	c.width = v.videoWidth || parseInt(v.style.width)
	const ctx = c.getContext('2d')
	ctx.drawImage(v, 0, 0)

	const back_img = new Image();
	back_img.onload = function() {
		let c = document.createElement('canvas')
		c.height = 640, c.width = 640
		const ctx = c.getContext('2d')

		var dx=0, dy=0, dWidth=0, dHeight=0;
		if (this.height >= this.width) {     // if its a tall image
			dWidth = (640 * this.width) / this.height
			dx = Math.floor((640-dWidth)/2)
			dy = 0
			dHeight = 640
		} else {									// if its a wide image
			dHeight = (640 * this.height) / this.width
			dx = 0
			dy = Math.floor((640-dHeight)/2)
			dWidth = 640
		}
		ctx.drawImage(this, dx, dy, dWidth, dHeight)
		
		$("#selectedImage").attr("src", c.toDataURL());
	}
	back_img.src = c.toDataURL()
	

});

let model;
$( document ).ready(async function () {
	tf.setBackend('webgl')
	console.log("backend: ", tf.getBackend())
	var startTime = (new Date()).getTime() 
	$('.progress-bar').html("Loading Model");
	$('.progress-bar').show();

    console.log( "Loading model..." );
	model = await tf.loadGraphModel('model/model.json');
	console.log( "Model loaded. Loading time: ", (new Date()).getTime() - startTime );

	startTime = (new Date()).getTime() 
	const warmupResult = model.predict(tf.zeros([1,640,640,3]));
	warmupResult.dispose();
	console.log("ms for warming up: ",(new Date()).getTime() - startTime)

	$('.progress-bar').hide();
});

function _logistic(x) {
	if (x > 0) {
	    return (1 / (1 + Math.exp(-x)));
	} else {
	    const e = Math.exp(x);
	    return e / (1 + e);
	}
}

async function loadImage() {
	console.log( "Pre-processing image..." );
	
	const pixels = $('#selectedImage').get(0);
		
	// Pre-process the image
	const img = tf.browser.fromPixels(pixels)
		.resizeNearestNeighbor([640,640])
		.toFloat()
		.div(tf.scalar(127.5))
		.expandDims();
	// const inp = tf.transpose(img, [0,3,1,2])
	return img;
}

var boxes_data, scores_data, classes_data;
$("#predict-button").click(async function () {
	$('.progress-bar').html("Starting prediction");
	$('.progress-bar').show();

	var startTime = (new Date()).getTime()
	const image = await loadImage();
	console.log("ms for image loading: ",(new Date()).getTime()-startTime)

	startTime = (new Date()).getTime()
	const res = model.execute(image); // inference model
	console.log("ms for model execution: ",(new Date()).getTime()-startTime)

	startTime = (new Date()).getTime()
	const transRes = res.transpose([0, 2, 1]); // transpose result [b, det, n] => [b, n, det]
	const boxes = tf.tidy(() => {
		const w = transRes.slice([0, 0, 2], [-1, -1, 1]); // get width
		const h = transRes.slice([0, 0, 3], [-1, -1, 1]); // get height
		const x1 = tf.sub(transRes.slice([0, 0, 0], [-1, -1, 1]), tf.div(w, 2)); // x1
		const y1 = tf.sub(transRes.slice([0, 0, 1], [-1, -1, 1]), tf.div(h, 2)); // y1
		return tf
		.concat(
			[
			x1,
			y1,
			tf.add(x1, w), //x2
			tf.add(y1, h), //y2
			],
			2
		)
		.squeeze();
	}); // process boxes [y1, child_x1, y2, x2]
	

	const [scores, classes] = tf.tidy(() => {
		// class scores
		const rawScores = transRes.slice([0, 0, 4], [-1, -1, 10]).squeeze(0); // #6 only squeeze axis 0 to handle only 1 class models
		return [rawScores.max(1), rawScores.argMax(1)];
	  }); // get max scores and classes index
	
	const nms = await tf.image.nonMaxSuppressionAsync(boxes, scores, 500, 0.45, 0.2); // NMS to filter boxes

	boxes_data = boxes.gather(nms, 0).arraySync(); // indexing boxes by nms index
	scores_data = scores.gather(nms, 0).arraySync(); // indexing scores by nms index
	classes_data = classes.gather(nms, 0).arraySync(); // indexing classes by nms index

	console.log("raw results: ", res.arraySync())
	console.log("boxes_data: ", boxes_data)
	console.log("scores_data: ", scores_data)
	console.log("classes_data: ", classes_data)

	console.log("ms for result processing: ",(new Date()).getTime()-startTime)

	startTime = (new Date()).getTime()
	const back_img = $('#selectedImage').get(0)
	const threshold = 0.45
	renderBoxes(back_img, threshold, boxes_data, scores_data, classes_data); // render boxes
	console.log("ms for box rendering: ", (new Date()).getTime()-startTime)
	
	tf.dispose([res, transRes, boxes, scores, classes, nms]); // clear memory
	$('.progress-bar').hide();
});
