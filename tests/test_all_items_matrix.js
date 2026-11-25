const fs = require('fs');
const html = fs.readFileSync('index.html', 'utf8');

const scriptStart = html.indexOf('<script>');
const scriptEnd = html.lastIndexOf('</script>');
let jsCode = html.substring(scriptStart + 8, scriptEnd);

jsCode += `
window.engineState = engineState;
window.computeDrinkPricingAndKb = computeDrinkPricingAndKb;
window.getFullDrinkTitle = getFullDrinkTitle;
window.processUtterance = processUtterance;
window.MENU_PAGES = MENU_PAGES;
window.getActiveItemRules = getActiveItemRules;
`;

const dom = {};
const mockElem = (id) => ({
  id, innerHTML: '', textContent: '', className: '', style: {},
  classList: { toggle: () => {}, add: () => {}, remove: () => {} },
  setAttribute: () => {}, getAttribute: () => '', querySelectorAll: () => [], addEventListener: () => {}
});
const document = {
  getElementById: (id) => dom[id] || (dom[id] = mockElem(id)),
  querySelectorAll: () => [], querySelector: () => null,
  body: { classList: { toggle: () => {} } }, addEventListener: () => {}
};
const window = { speechSynthesis: { speak: () => {}, cancel: () => {}, getVoices: () => [] } };
const vm = require('vm');
const ctx = { document, window, console, setTimeout, clearTimeout, setInterval, clearInterval, Float64Array, Math, Date, RegExp, String, Number, Array, Object };
vm.createContext(ctx);
vm.runInContext(jsCode, ctx);

const { engineState, computeDrinkPricingAndKb, getFullDrinkTitle, processUtterance } = ctx.window;

// Cross-Drink Transition Matrix (Multiple Zero Sugar Starting States)
const multiStartTests = [
  { startDrink: 'Teh', startMilk: 'None', startSweet: 'Zero Sugar', startTitle: 'Teh O Kosong', toUtterance: 'teh o', expSweet: 'Normal', expTitle: 'Teh O Panas' },
  { startDrink: 'Kopi', startMilk: 'Evaporated Milk', startSweet: 'Zero Sugar', startTitle: 'Kopi C Kosong', toUtterance: 'kopi c', expSweet: 'Normal', expTitle: 'Kopi C Panas' },
  { startDrink: 'Milo', startMilk: 'None', startSweet: 'Zero Sugar', startTitle: 'Milo O Kosong', toUtterance: 'milo o', expSweet: 'Normal', expTitle: 'Milo O Panas' },
  { startDrink: 'Americano', startMilk: 'None', startSweet: 'Zero Sugar', startTitle: 'Americano', toUtterance: 'kopi o', expSweet: 'Normal', expTitle: 'Kopi O Panas' },
  { startDrink: 'Cham', startMilk: 'None', startSweet: 'Zero Sugar', startTitle: 'Cham O Kosong', toUtterance: 'cham o', expSweet: 'Normal', expTitle: 'Cham O Panas' },
  { startDrink: 'Teh', startMilk: 'Evaporated Milk', startSweet: 'Zero Sugar', startTitle: 'Teh C Kosong', toUtterance: 'teh c', expSweet: 'Normal', expTitle: 'Teh C Panas' },
];

console.log('--- Testing Multiple Zero-Sugar Starting Drink Switches ---');
let allPassed = true;
for (const tc of multiStartTests) {
  engineState.base_drink = tc.startDrink;
  engineState.milk_type = tc.startMilk;
  engineState.sweetness_level = tc.startSweet;
  engineState.temperature = 'Hot';
  engineState.temp_detail = 'Hot';
  engineState.ice_level = null;
  engineState.status = 'AWAITING_CONFIRMATION';
  computeDrinkPricingAndKb();

  const reply = processUtterance(tc.toUtterance);
  const actualSweet = engineState.sweetness_level;
  const actualTitle = getFullDrinkTitle();
  const pass = actualSweet === tc.expSweet && actualTitle === tc.expTitle;
  if (!pass) allPassed = false;
  console.log(`[From "${tc.startTitle}" -> "${tc.toUtterance}"] -> Sweet: ${actualSweet} | Title: ${actualTitle} | ${pass ? 'PASS' : 'FAIL'}`);
}

if (!allPassed) {
  console.log('SOME MULTI-START TESTS FAILED');
  process.exit(1);
} else {
  console.log('ALL MULTI-START ZERO-SUGAR TESTS PASSED 100%!');
}
