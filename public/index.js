const showGlasses = document.getElementById('glasses');
const showHat = document.getElementById('hat');

// load the sprites
const glasses = new Image();
glasses.src = 'glasses.png';
const hat = new Image();
hat.src = 'hat.png';

// for alignment purposes, for each sprite we need to have the pixel location
// where the right eye (left on the image) goes. The other eye is assumed to
// be in a symmetrical location on the other half of the image.
const glassesAlign = {x: 80, y: 110};
const hatAlign = {x: 115, y: 245};

// the frame rate for face detection
// lower this setting to reduce CPU/GPU usage in the browser
const fps = 5;

const showLocalVideo = async () => {
  const frameInterval = 1000 / 5;

  let lastFaceTime = Date.now();
  let faces = null;

  // start a local video track
  const videoTrack = await Twilio.Video.createLocalVideoTrack({
    width: 640,
    height: 480,
    frameRate: 30,
  });
  document.getElementById('video').appendChild(videoTrack.attach());

  // align the canvas context with the eye locations returned by the face detector
  const align = (context, face) => {
    const rightEye = {x: face.landmarks[0][0], y: face.landmarks[0][1]};
    const leftEye = {x: face.landmarks[1][0], y: face.landmarks[1][1]};

    // the (0,0) point is going to be the center point between the eyes
    const origin = {x: (rightEye.x + leftEye.x) / 2, y: (rightEye.y + leftEye.y) / 2}
    context.translate(origin.x, origin.y);

    // calculate the angle of the line between the eyes with respect to the horizontal
    const angle = Math.atan2(origin.y - rightEye.y, origin.x - rightEye.x);
    context.rotate(angle);

    // return the distance between thw eyes, to be used in scaling the sprites
    return Math.sqrt((rightEye.x - leftEye.x) * (rightEye.x - leftEye.x) + (rightEye.y - leftEye.y) * (rightEye.y - leftEye.y));
  };

  // draw the sprite
  // the eyeDistance value returned by the align function is needed to compute
  // the scaling factor
  const drawSprite = (context, sprite, spriteAlign, eyeDistance) => {
    const scale = eyeDistance / (sprite.width - spriteAlign.x * 2);
    context.scale(scale, scale);
    context.drawImage(sprite, -sprite.width / 2, -spriteAlign.y)
    context.scale(1 / scale, 1 / scale);
  };

  // initialize tensorflow and blazeface
  await tf.setBackend('webgl');
  const model = await blazeface.load();

  // create a Twilio video processor
  const videoProcessor = {
    processFrame: async (input, output) => {
      const context = output.getContext('2d');
      if (context) {
        // draw the video image
        context.drawImage(input, 0, 0, input.width, input.height);

        const now = Date.now();
        if (!faces || now > lastFaceTime + frameInterval) {
          // find the faces
          const newFaces = await model.estimateFaces(input, false);
          if (newFaces) {
            faces = newFaces;
            lastFaceTime = now;
          }
        }
        if (faces.length) {
          // align the canvas with the face
          const eyeDistance = align(context, faces[0]);

          // draw the sprites on top of the image
          if (hat.complete && showHat.checked)
            drawSprite(context, hat, hatAlign, eyeDistance);
          if (glasses.complete && showGlasses.checked)
            drawSprite(context, glasses, glassesAlign, eyeDistance);

          // reset the transformation matrix
          context.resetTransform();
        }
      }
    },
  };

  // attach the video processor to the local video track
  videoTrack.addProcessor(videoProcessor);
}

showLocalVideo();
