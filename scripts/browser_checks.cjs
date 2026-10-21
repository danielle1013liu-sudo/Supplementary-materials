const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const out = path.resolve(process.env.BROWSER_EVIDENCE_DIR || 'docs/revision/browser');
const numerical = path.resolve(process.env.BROWSER_RESULTS_DIR || 'results/revision');
fs.mkdirSync(out,{recursive:true});
fs.mkdirSync(numerical,{recursive:true});
const release = JSON.parse(fs.readFileSync('data/publication/release_manifest.json','utf8'));
const bookVersion = release.publication_codebook_version;
assert.match(bookVersion,/^frequency_codes_[a-zA-Z0-9_]+$/);
const book = JSON.parse(fs.readFileSync(`data/public_codebooks/${bookVersion}.json`,'utf8'));
const models = JSON.parse(fs.readFileSync('models/manifest.json','utf8'));
const profile = JSON.parse(fs.readFileSync('results/revision/browser_case.json','utf8'));
const checks=[], anchors=[], browserErrors=[];
const appUrl=process.env.BROWSER_APP_URL||'http://127.0.0.1:8501';
function record(name){checks.push({check:name,status:'PASS'});console.log('PASS',name);}
async function open(context){
  const page=await context.newPage();
  page.on('pageerror',error=>browserErrors.push(error.message));
  page.on('console',entry=>{if(entry.type()==='error')browserErrors.push(entry.text());});
  await page.goto(appUrl);
  await page.locator('iframe').waitFor();
  const frame=await page.locator('iframe').contentFrame();
  await frame.getByRole('button',{name:'Model Prediction',exact:true}).waitFor({timeout:60000});
  return {page,frame};
}
async function checkImages(frame){
  const broken=await frame.locator('img').evaluateAll(async images=>{
    await Promise.all(images.map(image=>image.decode().catch(()=>{})));
    return images.filter(image=>!image.complete||image.naturalWidth===0).map(image=>image.alt);
  });
  assert.deepEqual(broken,[]);
}
async function captureHomeMobile(page,frame){
  await page.setViewportSize({width:390,height:844});
  await page.waitForFunction(()=>Math.abs(document.querySelector('iframe').getBoundingClientRect().height-window.innerHeight)<2);
  await checkImages(frame);
  assert.equal(await frame.locator('body').evaluate(el=>el.scrollWidth>el.clientWidth),false);
  await page.screenshot({path:path.join(out,'home_mobile_top.png')});
  const panel=frame.locator('.home-main');
  await panel.evaluate(el=>el.scrollTop=el.clientHeight*.65);
  await page.screenshot({path:path.join(out,'home_mobile_trees.png')});
  await panel.evaluate(el=>el.scrollTop=el.clientHeight*1.2);
  await page.screenshot({path:path.join(out,'home_mobile_delivery.png')});
  await panel.evaluate(el=>el.scrollTop=el.scrollHeight);
  await page.screenshot({path:path.join(out,'home_mobile_introduction.png')});
  await panel.evaluate(el=>el.scrollTop=0);
  await page.setViewportSize({width:1440,height:900});
  await page.waitForFunction(()=>Math.abs(document.querySelector('iframe').getBoundingClientRect().height-window.innerHeight)<2);
  record('desktop/mobile hero images decode and mobile Home has no horizontal overflow');
}
async function choose(frame,field,value){
  await frame.locator(`[data-choice="${field}"]`).click();
  await frame.locator(`[data-field="${field}"][data-value="${value}"]`).click();
}
async function capture(page,frame,name){
  await page.screenshot({path:path.join(out,name+'.png')});
  for(const [anchor,selector] of Object.entries({sidebar:'.sidebar',card:'.card',first_field:'.choice',slider:'.slider',predict:'.predict'})){
    const box=await frame.locator(selector).first().boundingBox();
    if(box)anchors.push({view:name,anchor,...box});
  }
}
async function input(frame,name,value){
  const field=frame.locator(`[data-number="${name}"]`);
  await field.fill(String(value));await field.press('Tab');
  assert.equal(await frame.locator(`[data-number="${name}"]`).inputValue(),String(value));
}
async function run(){
  const browser=await chromium.launch({headless:true});
  const context=await browser.newContext({viewport:{width:1440,height:900},deviceScaleFactor:1,acceptDownloads:true});
  context.setDefaultTimeout(20000);
  const {page,frame}=await open(context);
  assert.equal(await frame.locator('body').evaluate(el=>getComputedStyle(el).backgroundColor),'rgb(246, 240, 246)');
  assert.equal(await frame.locator('body').evaluate(el=>getComputedStyle(el).fontFamily),'Arial, Helvetica, sans-serif');
  record('original pale pink-lavender theme and Arial typography');
  assert.equal(await frame.locator('[data-page="tutorial"]').count(),0);
  assert.equal(await frame.getByRole('button',{name:'Tutorial',exact:true}).count(),0);
  record('Home navigation contains no Tutorial');
  await checkImages(frame);
  await page.screenshot({path:path.join(out,'home_1440.png')});
  await captureHomeMobile(page,frame);
  await frame.getByRole('button',{name:'Model Prediction',exact:true}).click();
  assert.equal(await frame.locator('.choice').count(),7);
  assert.equal(await frame.getByRole('button',{name:'Tutorial',exact:true}).count(),0);
  assert.equal(await frame.locator('.results').count(),0);record('navigation; seven categories; no invented initial results');
  await capture(page,frame,'prediction_initial_1440');
  await page.screenshot({path:path.join(out,'open_step.png')});
  for(const field of book.fields){
    await frame.locator(`[data-choice="${field.machine_column}"]`).click();
    assert.deepEqual(await frame.locator('.option').allTextContents(),field.categories.map(c=>`${c.new_code} · ${c.display_label}`));
    await frame.locator(`[data-field="${field.machine_column}"][data-value="${profile[field.machine_column]}"]`).click();
  }record('all dropdown orders and canonical choices');
  await page.screenshot({path:path.join(out,'categories_step.png')});
  await frame.getByRole('button',{name:'Collapse sidebar'}).click();
  assert.equal(Math.round((await frame.locator('.sidebar').boundingBox()).width),76);
  await frame.getByRole('button',{name:'Expand sidebar'}).click();record('sidebar collapse and expansion');
  await frame.getByRole('button',{name:'Open descriptor help'}).click();
  assert.equal(await frame.locator('.help-panel').count(),1);
  await frame.getByRole('button',{name:'Close descriptor help'}).click();record('descriptor help tab');
  for(const field of ['modulus_value_kPa','pore_size_um','culture_or_exposure_duration_h'])await input(frame,field,profile[field]);
  await input(frame,'modulus_value_kPa','1.2345678901234567');
  assert.equal(await frame.locator('[data-number="modulus_value_kPa"]').inputValue(),'1.2345678901234567');record('precise numeric entry preserved');
  await input(frame,'modulus_value_kPa','100000.123456789');
  assert.match(await frame.locator('#range-modulus_value_kPa').textContent(),/retained outside/);
  assert.equal(await frame.locator('[data-number="modulus_value_kPa"]').inputValue(),'100000.123456789');record('typed values outside slider range are not clamped');
  const slider=frame.locator('[data-slider="modulus_value_kPa"]');
  await slider.focus();await slider.press('Home');await slider.press('ArrowRight');await slider.press('Tab');
  assert.equal(Number(await frame.locator('[data-number="modulus_value_kPa"]').inputValue()),Number(await slider.inputValue()));record('slider synchronises the numeric field');
  await input(frame,'modulus_value_kPa',profile.modulus_value_kPa);
  await frame.locator('[data-missing="pore_size_um"]').check();
  assert.equal(await frame.locator('[data-number="pore_size_um"]').isDisabled(),true);
  await frame.locator('[data-missing="pore_size_um"]').uncheck();record('explicit unreported control');
  await frame.locator('.predict').evaluate(el=>el.scrollIntoView({block:"center"}));
  await capture(page,frame,'prediction_scrolled_1440');
  await page.screenshot({path:path.join(out,'numeric_step.png')});
  await input(frame,'modulus_value_kPa','not-a-number');
  await frame.getByRole('button',{name:'Predict',exact:true}).click();
  await frame.getByRole('alert').waitFor({timeout:60000});
  assert.match(await frame.getByRole('alert').innerText(),/must be a number/);record('malformed input is rejected');
  await input(frame,'modulus_value_kPa',profile.modulus_value_kPa);
  await frame.locator('.predict').evaluate(el=>el.scrollIntoView({block:"center"}));
  await page.screenshot({path:path.join(out,'predict_step.png')});
  await frame.getByRole('button',{name:'Predict',exact:true}).click();
  await frame.locator('.results').waitFor({timeout:60000});
  await frame.locator('.results').evaluate(el=>el.scrollIntoView({block:"center"}));
  assert.equal(await frame.locator('.result-card').count(),2);
  assert.equal(await frame.locator('.realisation').count(),6);
  for(const model of models)assert.match(await frame.locator('.results').innerText(),new RegExp(`Realisation ${model.seed}`));
  assert.match(await frame.locator('.results').innerText(),/TNF-α/);
  assert.match(await frame.locator('.results').innerText(),/pg\/mL/);record('two real target cards, all six predictions, units and SD');
  await capture(page,frame,'results_1440');
  await page.screenshot({path:path.join(out,'results_step.png')});
  const [download]=await Promise.all([page.waitForEvent('download'),frame.getByRole('link',{name:'Download predictions',exact:true}).click()]);
  await download.saveAs(path.join(numerical,'browser_predictions.csv'));
  const [inputDownload]=await Promise.all([page.waitForEvent('download'),frame.getByRole('link',{name:'Download inputs',exact:true}).click()]);
  await inputDownload.saveAs(path.join(numerical,'browser_inputs.csv'));
  const predictionCsv=fs.readFileSync(path.join(numerical,'browser_predictions.csv'),'utf8');
  for(const model of models)assert.match(predictionCsv,new RegExp(`${model.target},${model.seed},`));
  record('current six release realisations in CSV download');
  await frame.locator('.shap summary').click();
  assert.equal(await frame.locator('.shap-table tr').count(),20);record('grouped SHAP presentation');
  const separate=await browser.newContext({viewport:{width:1440,height:900}});
  const other=await open(separate);await other.frame.getByRole('button',{name:'Model Prediction',exact:true}).click();
  assert.equal(await other.frame.locator('.results').count(),0);
  assert.equal(await other.frame.locator('.choice.placeholder').count(),7);record('separate sessions do not share inputs or results');
  await separate.close();
  await input(frame,'modulus_value_kPa','2');
  assert.equal(await frame.locator('.results').count(),0);record('changing inputs invalidates stale results');
  await frame.locator('[data-missing="modulus_value_kPa"]').check();
  await frame.locator('[data-missing="pore_size_um"]').check();
  await frame.locator('[data-missing="culture_or_exposure_duration_h"]').check();
  await choose(frame,'macrophage_model_class','BMDM');
  await frame.getByRole('button',{name:'Predict',exact:true}).click();
  await frame.locator('.results').waitFor({timeout:60000});
  assert.match(await frame.locator('.results').innerText(),/unseen/);record('missing numeric inference and fitted-encoder warning');
  await frame.getByRole('button',{name:'Reset All',exact:true}).click();
  assert.equal(await frame.locator('.results').count(),0);assert.equal(await frame.locator('.choice.placeholder').count(),7);
  for(const field of ['modulus_value_kPa','pore_size_um','culture_or_exposure_duration_h'])assert.equal(await frame.locator(`[data-number="${field}"]`).inputValue(),'');
  await frame.locator('.reset').evaluate(el=>el.scrollIntoView({block:"center"}));
  await page.screenshot({path:path.join(out,'reset_step.png')});record('reset clears all inputs and results');
  assert.equal(await frame.locator('[data-page="tutorial"],.tutorial-body').count(),0);
  record('prediction sidebar contains no Tutorial page');
  await frame.getByRole('button',{name:'Landing / Home',exact:true}).click();
  for(const [width,height,name] of [[1875,1143,'reference_home'],[2048,1011,'reference_prediction_top'],[2048,1058,'reference_prediction_bottom'],[390,844,'mobile']]){
    await page.setViewportSize({width,height});
    await page.waitForFunction(() => Math.abs(document.querySelector('iframe').getBoundingClientRect().height - window.innerHeight) < 2);
    if(name==='reference_home')await page.screenshot({path:path.join(out,name+'.png')});
    else {
      if(await frame.getByRole('button',{name:'Model Prediction',exact:true}).count())await frame.getByRole('button',{name:'Model Prediction',exact:true}).click();
      if(name.endsWith('bottom'))await frame.locator('.predict').evaluate(el=>el.scrollIntoView({block:"center"}));
      else await frame.locator('.reset').evaluate(el=>el.scrollIntoView({block:"center"}));
      await capture(page,frame,name);
      const overflow=await frame.locator('body').evaluate(el=>el.scrollWidth>el.clientWidth);
      assert.equal(overflow,false);
      if(name==='mobile'){
        const boxes=await frame.locator('.choice').evaluateAll(nodes=>nodes.map(n=>n.getBoundingClientRect().x));
        assert.equal(new Set(boxes).size,1);
        for(const field of book.fields)await choose(frame,field.machine_column,profile[field.machine_column]);
        for(const field of ['modulus_value_kPa','pore_size_um','culture_or_exposure_duration_h'])await input(frame,field,profile[field]);
        await frame.getByRole('button',{name:'Predict',exact:true}).click();await frame.locator('.results').waitFor({timeout:60000});
        await frame.locator('.results').evaluate(el=>el.scrollIntoView({block:"center"}));
        const cards=await frame.locator('.result-card').evaluateAll(nodes=>nodes.map(n=>({x:n.getBoundingClientRect().x,y:n.getBoundingClientRect().y})));
        assert.equal(cards[0].x,cards[1].x);assert.ok(cards[1].y>cards[0].y);
        await capture(page,frame,'mobile_results');record('390×844 stacked form and results without horizontal overflow');
      }
    }
  }
  record('1440×900 and supplied-reference-size captures');
  fs.writeFileSync(path.join(out,'browser_errors.json'),JSON.stringify(browserErrors,null,2));
  assert.deepEqual(browserErrors,[]);
  record('no browser JavaScript or console errors');
  fs.writeFileSync(path.join(numerical,'browser_checks.csv'),'check,status\n'+checks.map(c=>`"${c.check.replaceAll('"','""')}",${c.status}`).join('\n')+'\n');
  fs.writeFileSync(path.join(numerical,'browser_anchors.csv'),'view,anchor,x,y,width,height\n'+anchors.map(a=>[a.view,a.anchor,a.x,a.y,a.width,a.height].join(',')).join('\n')+'\n');
  fs.rmSync(path.join(numerical,'browser_failure.txt'),{force:true});
  await browser.close();
}
run().catch(error=>{fs.writeFileSync(path.join(numerical,'browser_failure.txt'),error.stack);console.error(error);process.exit(1);});
