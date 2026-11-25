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

// Check all 13 items ordered with only temperature & sweetness (e.g. 'kopi panas normal')
// Verify that NONE of them ask 'ask_milk' (e.g. 'Susu Pekat, Susu Cair')
const itemsToTest = [
  { utt: 'kopi panas normal', expTitle: 'Kopi Panas', expMilk: 'Condensed Milk' },
  { utt: 'kopi o panas normal', expTitle: 'Kopi O Panas', expMilk: 'None' },
  { utt: 'kopi c panas normal', expTitle: 'Kopi C Panas', expMilk: 'Evaporated Milk' },
  { utt: 'teh panas normal', expTitle: 'Teh Panas', expMilk: 'Condensed Milk' },
  { utt: 'teh o panas normal', expTitle: 'Teh O Panas', expMilk: 'None' },
  { utt: 'teh c panas normal', expTitle: 'Teh C Panas', expMilk: 'Evaporated Milk' },
  { utt: 'milo panas normal', expTitle: 'Milo Panas', expMilk: 'Condensed Milk' },
  { utt: 'milo o panas normal', expTitle: 'Milo O Panas', expMilk: 'None' },
  { utt: 'cham panas normal', expTitle: 'Cham Panas', expMilk: 'Condensed Milk' },
  { utt: 'white coffee panas normal', expTitle: 'White Coffee Panas', expMilk: 'Condensed Milk' },
  { utt: 'neslo panas normal', expTitle: 'Neslo Panas', expMilk: 'Condensed Milk' },
  { utt: 'teh o limau panas normal', expTitle: 'Teh O Limau Panas', expMilk: 'None' },
  { utt: 'americano panas kosong', expTitle: 'Hot Americano (Kosong / No Sugar)', expMilk: 'None' },
];

console.log('--- Checking all 13 menu items for unwanted ask_milk question ---');
let allPassed = true;
for (const tc of itemsToTest) {
  // Reset state
  engineState.selected_item_id = null;
  engineState.base_drink = null;
  engineState.milk_type = null;
  engineState.sweetness_level = null;
  engineState.temperature = null;
  engineState.temp_detail = null;
  engineState.ice_level = null;
  engineState.status = 'IDLE';
  engineState.pending_slot = null;

  const reply = processUtterance(tc.utt);
  const askedMilk = reply.includes('Susu Pekat') || reply.includes('ask_milk') || engineState.pending_slot === 'milk_type';
  const title = getFullDrinkTitle();
  const milk = engineState.milk_type || 'None';
  const pass = !askedMilk && milk === tc.expMilk && title === tc.expTitle;
  if (!pass) allPassed = false;
  console.log(`[Order: "${tc.utt.padEnd(28)}"] -> Title: ${title.padEnd(35)} | Milk: ${milk.padEnd(16)} | Asked Milk?: ${askedMilk ? 'YES (FAIL)' : 'NO (PASS)'}`);
}

console.log('--- Result ---');
if (allPassed) {
  console.log('SUCCESS: All 13 items have locked milk and NEVER ask redundant milk questions!');
} else {
  console.log('FAILURE: Some items still asked milk!');
  process.exit(1);
}
