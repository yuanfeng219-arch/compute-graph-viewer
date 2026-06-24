(function(global){
  const FORBIDDEN_HUE_RANGES=[
    {name:'pure-red',from:350,to:14,wraps:true},
    {name:'pure-green',from:104,to:146,wraps:false}
  ];

  const CATEGORY_HUES={
    dense:214,
    embedding:226,
    attention:198,
    norm:176,
    residual:192,
    router:282,
    expert:36,
    comm:318,
    head:326,
    moe:282,
    rank0:36,
    rank7:192,
    rank16:318,
    neutral:226
  };

  const RANK_HUES=[
    36,46,58,72,168,184,198,212,
    226,240,254,268,282,296,312,326,
    28,42,56,176,190,204,218,232,
    246,260,274,288,304,320,336,66
  ];

  const VARIANT_OFFSETS=[0,-8,8,-14,14,-20,20];
  const VARIANT_LIGHT_OFFSETS=[0,.015,-.012,.024,-.018,.032,-.024];
  const VARIANT_SAT_FACTORS=[1,.92,.96,.86,.90,.82,.86];

  const SEMANTIC_STYLE={
    'sem:embedding':{category:'embedding',variant:0,s:.86,l:.01},
    'module:decoder':{category:'dense',variant:0,s:1,l:0},
    'sem:attention':{category:'attention',variant:0,s:1.02,l:.012},
    'sem:norm':{category:'norm',variant:0,s:.74,l:.024},
    'sem:residual':{category:'residual',variant:1,s:.78,l:.018},
    'sem:head':{category:'head',variant:0,s:.84,l:.012},
    'module:model':{category:'neutral',variant:0,s:.46,l:.02},

    'sem:gate':{category:'router',variant:0,s:1.02,l:.005},
    'sem:comm':{category:'comm',variant:0,s:.92,l:.014},
    'sem:moe':{category:'expert',variant:0,s:1,l:0},
    'sem:mlp':{category:'expert',variant:1,s:.88,l:.014},
    'module:moe':{category:'moe',variant:0,s:1,l:0},
    'module:mlp':{category:'expert',variant:1,s:.88,l:.014},

    'io:input':{category:'neutral',variant:1,s:.46,l:.02},
    'io:parameter':{category:'neutral',variant:3,s:.40,l:.03},
    'io:output':{category:'neutral',variant:2,s:.50,l:.03},
    'pipeline:bubble':{category:'neutral',variant:5,s:.30,l:.01}
  };

  const PASS_IR_DARK_ANCHORS={
    semantic:{
      'module:decoder':'#5192ff',
      'sem:embedding':'#6b92ff',
      'sem:attention':'#49c5f6',
      'sem:norm':'#d8b900',
      'sem:residual':'#87c80f',
      'sem:head':'#c9107d',
      'module:model':'#64748b',

      'sem:gate':'#a855f7',
      'sem:comm':'#c9107d',
      'sem:moe':'#fa8c42',
      'sem:mlp':'#f6b24d',
      'module:moe':'#a855f7',
      'module:mlp':'#fa8c42',

      'io:input':'#6b92ff',
      'io:parameter':'#f9823a',
      'io:output':'#49c5f6',
      'pipeline:bubble':'#3b3b3b'
    },
    category:{
      dense:['#5192ff','#4a8fff','#6b92ff','#3577f6'],
      embedding:['#6b92ff','#5192ff','#7aa2ff'],
      attention:['#49c5f6','#38bdf8','#6bd5ff'],
      norm:['#d8b900','#e1c84a','#c8ae0f'],
      residual:['#87c80f','#a6d92c','#d8b900'],
      router:['#a855f7','#b86cff','#9457e9'],
      expert:['#fa8c42','#f9823a','#f6b24d','#d45a08'],
      comm:['#c9107d','#e052b0','#d946ef'],
      head:['#c9107d','#e0529a','#ff6fae'],
      moe:['#a855f7','#9457e9','#b86cff'],
      rank0:['#fa8c42','#f6b24d','#d8b900','#f9823a'],
      rank7:['#49c5f6','#6b92ff','#5192ff','#38bdf8'],
      rank16:['#c9107d','#a855f7','#e052b0','#9457e9'],
      neutral:['#64748b','#7a8494','#8b929e']
    },
    rank:[
      '#fa8c42','#f6b24d','#d8b900','#87c80f',
      '#49c5f6','#6b92ff','#5192ff','#3577f6',
      '#a855f7','#9457e9','#c9107d','#e052b0',
      '#f9823a','#e1c84a','#a6d92c','#38bdf8'
    ]
  };

  const PASS_IR_LIGHT_ANCHORS={
    semantic:{
      'module:decoder':'#9bbae6',
      'sem:embedding':'#a4b0dd',
      'sem:attention':'#8dcfd5',
      'sem:norm':'#e6db92',
      'sem:residual':'#c1dc75',
      'sem:head':'#beaee1',
      'module:model':'#85909e',

      'sem:gate':'#dab8e2',
      'sem:comm':'#beaee1',
      'sem:moe':'#e6b696',
      'sem:mlp':'#e6db92',
      'module:moe':'#dab8e2',
      'module:mlp':'#e6b696',

      'io:input':'#a4b0dd',
      'io:parameter':'#e6b696',
      'io:output':'#8dcfd5',
      'pipeline:bubble':'#b7c0cb'
    },
    category:{
      dense:['#9bbae6','#8fb0dd','#a4b0dd','#84a7d6'],
      embedding:['#a4b0dd','#9bbae6','#b0b8de'],
      attention:['#8dcfd5','#80c6ce','#9ed5da'],
      norm:['#e6db92','#dccf72','#dfd59d'],
      residual:['#c1dc75','#b3d264','#cbdf91'],
      router:['#dab8e2','#d1aadc','#beaee1'],
      expert:['#e6b696','#dcaa89','#e6db92','#d29c76'],
      comm:['#beaee1','#b3a0d9','#dab8e2'],
      head:['#beaee1','#b3a0d9','#dab8e2'],
      moe:['#dab8e2','#d1aadc','#beaee1'],
      rank0:['#e6b696','#e6db92','#c1dc75','#dcaa89'],
      rank7:['#8dcfd5','#9bbae6','#a4b0dd','#80c6ce'],
      rank16:['#beaee1','#dab8e2','#b3a0d9','#d1aadc'],
      neutral:['#85909e','#9199a5','#9ba2ac']
    },
    rank:[
      '#9bbae6','#a4b0dd','#8dcfd5','#dab8e2',
      '#e6b696','#beaee1','#e6db92','#c1dc75',
      '#9bbae6','#a4b0dd','#8dcfd5','#dab8e2',
      '#e6b696','#beaee1','#e6db92','#c1dc75'
    ]
  };

  const PALETTE_PROFILES=[
    {
      id:'balanced',
      name:'Pass-IR Contrast',
      note:'Dark anchors borrowed from pass-ir: clean blue, orange, gold, violet and magenta.',
      hues:{dense:214,embedding:226,attention:198,norm:176,residual:192,router:282,expert:36,comm:318,head:326,moe:282,rank0:36,rank7:192,rank16:318,neutral:226},
      rankHues:[
        36,46,58,72,168,184,198,212,
        226,240,254,268,282,296,312,326,
        28,42,56,176,190,204,218,232,
        246,260,274,288,304,320,336,66
      ],
      dark:{s:.78,l:.55},
      light:{s:.50,l:.70},
      darkAnchors:PASS_IR_DARK_ANCHORS,
      lightAnchors:PASS_IR_LIGHT_ANCHORS
    },
    {
      id:'bright',
      name:'Cool Hardware',
      note:'Cyan-leaning Dense, violet MoE, cooler rank ladder.',
      hues:{dense:198,embedding:220,attention:188,norm:166,residual:202,router:264,expert:28,comm:304,head:328,moe:264,rank0:28,rank7:178,rank16:304,neutral:222},
      rankHues:[
        28,40,52,64,176,188,200,212,
        224,236,248,260,272,284,296,308,
        320,332,24,36,184,196,208,220,
        232,244,256,268,280,292,316,340
      ],
      dark:{s:.86,l:.58},
      light:{s:.56,l:.71}
    },
    {
      id:'crisp',
      name:'Warm Ranks',
      note:'Royal-blue Dense, magenta-purple MoE, warmer rank emphasis.',
      hues:{dense:228,embedding:238,attention:208,norm:180,residual:194,router:294,expert:50,comm:330,head:316,moe:294,rank0:50,rank7:208,rank16:330,neutral:238},
      rankHues:[
        24,34,44,54,64,74,166,180,
        194,208,222,236,250,264,278,292,
        306,320,334,28,40,52,68,176,
        190,204,218,232,246,260,286,314
      ],
      dark:{s:.84,l:.52},
      light:{s:.52,l:.69}
    },
    {
      id:'soft',
      name:'Soft Split',
      note:'Softer blue Dense, lavender MoE, pastel rank families.',
      hues:{dense:206,embedding:218,attention:190,norm:166,residual:178,router:274,expert:64,comm:306,head:326,moe:274,rank0:64,rank7:170,rank16:306,neutral:214},
      rankHues:[
        64,52,40,28,170,182,194,206,
        218,230,242,254,266,278,290,302,
        314,326,338,58,46,34,166,178,
        190,202,214,226,238,250,274,298
      ],
      dark:{s:.64,l:.56},
      light:{s:.38,l:.74}
    }
  ];

  // light 取色：低饱和 + 中高明度（柔和但不发白）。饱和度低于 dark，明度高于 dark。
  const LIGHT_VARIANTS=[
    {
      id:'clear',
      name:'Clean Pastel',
      note:'低饱和、中高明度的干净 pastel，白底上清爽但不发白。',
      curve:{s:.48,l:.72}
    },
    {
      id:'fresh',
      name:'Fresh Pastel',
      note:'pastel 里色度最足的一档，仍明显低于 dark 饱和。',
      curve:{s:.56,l:.70}
    },
    {
      id:'paper',
      name:'Soft Pastel',
      note:'最柔最浅，纸感，弱对比场景用。',
      curve:{s:.40,l:.76}
    },
    {
      id:'crisp-light',
      name:'Crisp Pastel',
      note:'明度稍收、色相分离更清楚，仍是高明度低饱和。',
      curve:{s:.52,l:.69}
    }
  ];

  function clamp(v,min,max){return Math.max(min,Math.min(max,v));}

  function isForbiddenHue(deg){
    const h=((deg%360)+360)%360;
    return FORBIDDEN_HUE_RANGES.some(range=>{
      if(range.wraps)return h>=range.from||h<=range.to;
      return h>=range.from&&h<=range.to;
    });
  }

  function circularDistance(a,b){
    const d=Math.abs(a-b)%360;
    return Math.min(d,360-d);
  }

  function snapHue(deg){
    let h=((deg%360)+360)%360;
    if(!isForbiddenHue(h))return h;
    const candidates=[];
    FORBIDDEN_HUE_RANGES.forEach(range=>{
      candidates.push((range.from+360-8)%360,(range.to+8)%360);
    });
    candidates.sort((a,b)=>circularDistance(h,a)-circularDistance(h,b));
    return candidates.find(c=>!isForbiddenHue(c))??36;
  }

  function hslToHex({h,s,l}){
    const hue=(p,q,t)=>{
      let x=t;
      if(x<0)x+=1;
      if(x>1)x-=1;
      if(x<1/6)return p+(q-p)*6*x;
      if(x<1/2)return q;
      if(x<2/3)return p+(q-p)*(2/3-x)*6;
      return p;
    };
    let r,g,b;
    if(s===0){r=g=b=l;}
    else{
      const q=l<.5?l*(1+s):l+s-l*s;
      const p=2*l-q;
      r=hue(p,q,h+1/3);
      g=hue(p,q,h);
      b=hue(p,q,h-1/3);
    }
    return '#'+[r,g,b].map(v=>Math.round(clamp(v,0,1)*255).toString(16).padStart(2,'0')).join('');
  }

  function hashHue(key,index){
    let hash=2166136261;
    const text=String(key);
    for(let i=0;i<text.length;i++){
      hash^=text.charCodeAt(i);
      hash=Math.imul(hash,16777619);
    }
    const warm=[24,34,44,54,64,316,328,340];
    const cool=[166,178,190,202,214,226,238,250,262,274,286,298];
    const pool=index%3===0?warm.concat(cool):cool.concat(warm);
    return pool[(hash>>>0)%pool.length];
  }

  function rankHue(rank,profile){
    const n=Number(rank);
    const hues=profile?.rankHues||RANK_HUES;
    const base=profile?.hues||CATEGORY_HUES;
    if(Number.isFinite(n)&&n>=0)return snapHue(hues[Math.floor(n)%hues.length]);
    return base.rank0;
  }

  function styleForKey(key,index){
    if(SEMANTIC_STYLE[key])return SEMANTIC_STYLE[key];
    if(key==='cat:dense')return {category:'dense',variant:0,s:1,l:0};
    if(key==='cat:moe')return {category:'moe',variant:0,s:1,l:0};
    const rank=String(key).match(/^rank:r(\d+)(?::v(\d+))?$/);
    if(rank)return {category:'rank',rank:Number(rank[1]),variant:Number(rank[2]||0),s:1,l:0};
    const cat=String(key).match(/^cat:(dense|moe|rank0|rank7|rank16)(?::v(\d+))?$/);
    if(cat)return {category:cat[1],variant:Number(cat[2]||0),s:1,l:0};
    return {category:'custom',hue:hashHue(key,index),variant:index||0,s:.82,l:0};
  }

  function baseHueForStyle(style,profile){
    const hues=profile?.hues||CATEGORY_HUES;
    if(style.category==='rank')return rankHue(style.rank,profile);
    if(style.category==='rank0')return hues.rank0;
    if(style.category==='rank7')return hues.rank7;
    if(style.category==='rank16')return hues.rank16;
    if(style.category==='custom')return style.hue;
    return hues[style.category]??hues.neutral??CATEGORY_HUES.neutral;
  }

  function profileById(id){
    return PALETTE_PROFILES.find(p=>p.id===id)||PALETTE_PROFILES[0];
  }

  function lightVariantById(id){
    return LIGHT_VARIANTS.find(v=>v.id===id)||LIGHT_VARIANTS[0];
  }

  function lightCurveForProfile(profile,lightVariantId){
    const v=lightVariantById(lightVariantId);
    return {
      s:clamp(Math.min(v.curve.s,profile.dark.s*.85),.22,.62),  // 低饱和：必定 < dark
      l:clamp(Math.max(v.curve.l,profile.dark.l+.10),.64,.82)   // 明度高于 dark，但不再自动抬到发白
    };
  }

  function anchoredProfileColor(key,style,profile,mode){
    const anchors=mode==='light'?profile?.lightAnchors:profile?.darkAnchors;
    if(!anchors)return null;
    if(anchors.semantic?.[key])return anchors.semantic[key];
    const variant=Math.abs(style.variant||0);
    if(style.category==='rank'&&Array.isArray(anchors.rank)&&anchors.rank.length){
      const rank=Number.isFinite(style.rank)?Math.floor(style.rank):0;
      return anchors.rank[(rank*3+variant)%anchors.rank.length];
    }
    const colors=anchors.category?.[style.category];
    if(Array.isArray(colors)&&colors.length)return colors[variant%colors.length];
    return null;
  }

  function colorFromStyle(key,style,profile,theme,lightVariantId){
    const mode=theme==='light'?'light':'dark';
    const anchored=anchoredProfileColor(key,style,profile,mode);
    if(anchored)return anchored;
    const curve=mode==='light'?lightCurveForProfile(profile,lightVariantId):(profile.dark);
    const variant=Math.abs(style.variant||0);
    const hue=snapHue(baseHueForStyle(style,profile)+(VARIANT_OFFSETS[variant%VARIANT_OFFSETS.length]||0));
    const lightBoost=mode==='light'?Math.abs(VARIANT_LIGHT_OFFSETS[variant%VARIANT_LIGHT_OFFSETS.length]||0):VARIANT_LIGHT_OFFSETS[variant%VARIANT_LIGHT_OFFSETS.length]||0;  // light 偏移取正→更亮
    return hslToHex({
      h:hue/360,
      s:clamp(curve.s*(style.s??1)*(VARIANT_SAT_FACTORS[variant%VARIANT_SAT_FACTORS.length]||1),.18,.90),
      l:clamp(curve.l+(style.l||0)+lightBoost,.34,.90)
    });
  }

  function buildPanguSemanticColors({keys,paletteId,theme,lightVariantId}){
    const profile=profileById(paletteId);
    const list=[...new Set(keys||Object.keys(SEMANTIC_STYLE))];
    return Object.fromEntries(list.map((key,index)=>[key,colorFromStyle(key,styleForKey(key,index),profile,theme,lightVariantId)]));
  }

  function getPanguCategoryColor(category,variant,{paletteId,theme,lightVariantId}={}){
    const profile=profileById(paletteId);
    const key=`cat:${category}:v${variant||0}`;
    const style=styleForKey(key,variant||0);
    return colorFromStyle(key,style,profile,theme,lightVariantId);
  }

  global.PANGU_FORBIDDEN_HUE_RANGES=FORBIDDEN_HUE_RANGES.map(r=>({...r}));
  global.PANGU_CATEGORY_HUES={...CATEGORY_HUES};
  global.PANGU_PALETTE_PROFILES=PALETTE_PROFILES;
  global.PANGU_LIGHT_VARIANTS=LIGHT_VARIANTS;
  global.getPanguPaletteProfiles=()=>PALETTE_PROFILES.map(p=>({...p,dark:{...p.dark},light:{...p.light},hues:{...p.hues},rankHues:[...p.rankHues],darkAnchors:p.darkAnchors,lightAnchors:p.lightAnchors}));
  global.getPanguLightVariants=()=>LIGHT_VARIANTS.map(v=>({...v,curve:{...v.curve}}));
  global.getPanguCategoryColor=getPanguCategoryColor;
  global.buildPanguSemanticColors=buildPanguSemanticColors;
})(window);
