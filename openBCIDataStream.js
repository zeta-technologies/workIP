const Ganglion = require('./OBGanglion/index').Ganglion;
const ganglion = new Ganglion();
var origin = 0;
var cpt = 0;
var start = new Date().getTime();

ganglion.once('ganglionFound', (peripheral) => {
  // Stop searching for BLE devices once a ganglion is found.
  ganglion.searchStop();
  ganglion.on('sample', (sample) => {
    /** Work with sample */

    console.log(sample.sampleNumber);
    for (let i = 0; i < ganglion.numberOfChannels(); i++) {
      console.log(sample.channelData[i].toFixed(8)); //+ " Volts.");
    }
  });
  ganglion.once('ready', () => {
    ganglion.streamStart();
  });
  ganglion.connect(peripheral);
});
// Start scanning for BLE devices
ganglion.searchStart();
