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
  id,
  innerHTML: '',
  textContent: '',
  className: '',
  style: {},
  classList: { toggle: () => {}, add: () => {}, remove: () => {} },
  setAttribute: () => {},
  getAttribute: () => '',
  querySelectorAll: () => [],
  addEventListener: () => {}
});

const document = {
  getElementById: (id) => dom[id] || (dom[id] = mockElem(id)),
  querySelectorAll: () => [],
  querySelector: () => null,
  body: { classList: { toggle: () => {} } },
  addEventListener: () => {}
};
const window = {
  speechSynthesis: { speak: () => {}, cancel: () => {}, getVoices: () => [] }
};

const vm = require('vm');
const ctx = { document, window, console, setTimeout, clearTimeout, setInterval, clearInterval, Float64Array, Math, Date, RegExp, String, Number, Array, Object };
vm.createContext(ctx);
vm.runInContext(jsCode, ctx);

const { engineState, computeDrinkPricingAndKb, getFullDrinkTitle, processUtterance, MENU_PAGES, getActiveItemRules } = ctx.window;

console.log('=== TEST 1: Select Cham Ice and send "change to no sugar" ===');
engineState.selected_item_id = 'cham';
engineState.base_drink = 'Cham';
engineState.temperature = 'Iced';
engineState.milk_type = 'Condensed Milk';
engineState.sweetness_level = 'Normal';
engineState.status = 'AWAITING_CONFIRMATION';
computeDrinkPricingAndKb();

console.log('Initial:', getFullDrinkTitle(), '| Sweetness:', engineState.sweetness_level);
let r1 = processUtterance('change to no sugar');
console.log('Bot Reply:', r1);
console.log('Resulting Title:', getFullDrinkTitle());
console.log('Resulting Sweetness:', engineState.sweetness_level);
console.log('Resulting Calories & Sugar:', engineState.calories, 'kcal,', engineState.sugars_g, 'g sugar');
if (engineState.sweetness_level !== 'Less Sugar') throw new Error('Cham sweetness must be Less Sugar!');
if (getFullDrinkTitle().includes('Kosong')) throw new Error('Cham title must NOT contain Kosong!');

console.log('\n=== TEST 2: Select Cham Ice and send "no sugar" directly ===');
engineState.selected_item_id = 'cham';
engineState.base_drink = 'Cham';
engineState.temperature = 'Iced';
engineState.milk_type = 'Condensed Milk';
engineState.sweetness_level = 'Normal';
engineState.status = 'AWAITING_CONFIRMATION';
computeDrinkPricingAndKb();

let r2 = processUtterance('no sugar');
console.log('Bot Reply:', r2);
console.log('Resulting Title:', getFullDrinkTitle());
console.log('Resulting Sweetness:', engineState.sweetness_level);
if (engineState.sweetness_level !== 'Less Sugar') throw new Error('Cham sweetness must be Less Sugar!');

console.log('\n=== TEST 3: Check All 13 Menu Items for "no sugar" ===');
const allItems = MENU_PAGES.flatMap(p => p.items);
allItems.forEach(item => {
  engineState.selected_item_id = item.id;
  engineState.base_drink = item.base;
  engineState.temperature = 'Iced';
  engineState.milk_type = item.defaultMilk;
  engineState.sweetness_level = 'Normal';
  engineState.status = 'AWAITING_CONFIRMATION';
  computeDrinkPricingAndKb();
  
  const rules = getActiveItemRules();
  processUtterance('no sugar');
  
  if (!rules.allowZeroSugar) {
    console.log(`[DISALLOW ZERO] ${item.id.padEnd(14)} (${item.defaultMilk.padEnd(14)}): Sweetness -> ${engineState.sweetness_level} (PASS: clamped to Less Sugar)`);
    if (engineState.sweetness_level !== 'Less Sugar') throw new Error(`${item.id} should be clamped to Less Sugar`);
  } else {
    console.log(`[ALLOW ZERO]    ${item.id.padEnd(14)} (${item.defaultMilk.padEnd(14)}): Sweetness -> ${engineState.sweetness_level} (PASS: Zero Sugar allowed)`);
    if (engineState.sweetness_level !== 'Zero Sugar') throw new Error(`${item.id} should allow Zero Sugar`);
  }
});

console.log('\n=== TEST 4: Health Query Q&A Verification ===');
let r4 = processUtterance('Ada drink apa yg zero sugar and low calories ah?');
console.log('Bot Reply:', r4);
if (!r4.includes('Kopi O Kosong') || !r4.includes('Teh O Kosong')) throw new Error('Must return low-calorie recommendations');

console.log('\n=== TEST 5: Kopi O Kosong to Kopi O Switch (Sugar Level & Title) ===');
engineState.base_drink = 'Kopi';
engineState.milk_type = 'None';
engineState.sweetness_level = 'Zero Sugar';
engineState.temperature = 'Hot';
engineState.status = 'AWAITING_CONFIRMATION';
computeDrinkPricingAndKb();
console.log('Initial Drink:', getFullDrinkTitle(), '| Sweetness:', engineState.sweetness_level);

let r5 = processUtterance('kopi o');
console.log('Bot Reply after "kopi o":', r5);
console.log('Resulting Drink:', getFullDrinkTitle());
console.log('Resulting Sweetness:', engineState.sweetness_level);
if (engineState.sweetness_level !== 'Normal') throw new Error('Kopi O must have Normal sugar!');
if (getFullDrinkTitle().includes('Kosong')) throw new Error('Kopi O title must NOT contain Kosong!');
if (!getFullDrinkTitle().includes('Kopi O')) throw new Error('Title must contain Kopi O!');

console.log('\n=== TEST 6: User says "i want kopi o 0" ===');
engineState.base_drink = null;
engineState.milk_type = null;
engineState.sweetness_level = null;
engineState.temperature = null;
let r6 = processUtterance('i want kopi o 0');
console.log('Bot Reply after "i want kopi o 0":', r6);
console.log('Resulting Sweetness:', engineState.sweetness_level, '| Milk:', engineState.milk_type);
if (engineState.sweetness_level !== 'Zero Sugar') throw new Error('0 must be recognized as Zero Sugar!');
if (engineState.milk_type !== 'None') throw new Error('Kopi O must have No Milk (None)!');

console.log('\n=== ALL TESTS PASSED SUCCESSFULLY! ===');
