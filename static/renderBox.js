
export const renderBoxes = (back_img, threshold, boxes_data, scores_data, classes_data) => {

  const c = document.createElement('canvas')
  const ctx = c.getContext("2d");
  ctx.canvas.width=640
  ctx.canvas.height=640

  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height); // clean canvas

  var dx=0, dy=0, dWidth=0, dHeight=0;
  if (back_img.height >= back_img.width) {     // if its a tall image
    dWidth = (640 * back_img.width) / back_img.height
    dx = Math.floor((640-dWidth)/2)
    dy = 0
    dHeight = 640
  } else {									// if its a wide image
    dHeight = (640 * back_img.height) / back_img.width
    dx = 0
    dy = Math.floor((640-dHeight)/2)
    dWidth = 640
  }
  ctx.drawImage(back_img, dx, dy, dWidth, dHeight)

  // font configs
  const font = "12px sans-serif";
  ctx.font = font;
  ctx.textBaseline = "top";

  for (let i = 0; i < scores_data.length; ++i) {
    //console.log('scores_data[i]: ', scores_data[i])
    if (scores_data[i] > threshold) {
      const klass = TARGET_CLASSES[classes_data[i]];
      const score = (scores_data[i] * 100).toFixed(1);

      let [x1, y1, x2, y2] = boxes_data[i]
      const width = x2 - x1;
      const height = y2 - y1;

      // Draw the bounding box.
      ctx.strokeStyle = "#B033FF";
      ctx.lineWidth = 2;
      ctx.strokeRect(x1, y1, width, height);

      // Draw the label background.
      ctx.fillStyle = "#B033FF";
      const textWidth = ctx.measureText(klass).width;
      const textHeight = parseInt(font, 10); // base 10
      ctx.fillRect(Math.max(x1 - 1, x1-1+(width-textWidth)/2), y1 - (textHeight + 2), textWidth + 2, textHeight + 2);

      // Draw labels
      ctx.fillStyle = "#ffffff";
      ctx.fillText(klass, Math.max(x1 - 1, x1-1+(width-textWidth)/2), y1 - (textHeight + 2));
    }
  }
  var outputImageURL = c.toDataURL()
  $("#outputImage").attr("src", outputImageURL);
};
