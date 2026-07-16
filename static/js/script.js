/* =============================================================================
   script.js — YentiFlix
   Vanilla JavaScript — no jQuery, no frameworks.
   Runs after DOM is ready via DOMContentLoaded.
============================================================================= */

(function () {
  'use strict';

  /* ═══════════════════════════════════════════════════════════════════════
     UTILITY HELPERS
  ════════════════════════════════════════════════════════════════════════ */

  /** Select a single element — returns null safely */
  function $(selector, parent) {
    return (parent || document).querySelector(selector);
  }

  /** Select all elements — returns a real Array */
  function $$(selector, parent) {
    return Array.from((parent || document).querySelectorAll(selector));
  }

  /** Debounce — limits how often fn fires (e.g. on scroll/resize) */
  function debounce(fn, wait) {
    var t;
    return function () {
      clearTimeout(t);
      t = setTimeout(fn, wait);
    };
  }

  /* ═══════════════════════════════════════════════════════════════════════
     1. SMOOTH SCROLLING
     Intercepts clicks on any <a href="#..."> anchor and scrolls smoothly
     to the target element instead of jumping instantly.
  ════════════════════════════════════════════════════════════════════════ */

  function initSmoothScroll() {
    $$('a[href^="#"]').forEach(function (link) {
      link.addEventListener('click', function (e) {
        var hash  = this.getAttribute('href');
        if (hash === '#') return; // Skip bare # links
        var target = $(hash);
        if (!target) return;
        e.preventDefault();
        var navH   = parseInt(
          getComputedStyle(document.documentElement).getPropertyValue('--nav-h') || '66',
          10
        );
        var top = target.getBoundingClientRect().top + window.pageYOffset - navH - 12;
        window.scrollTo({ top: top, behavior: 'smooth' });
        // Update URL hash without jumping
        if (history.pushState) history.pushState(null, null, hash);
      });
    });
  }

  /* ═══════════════════════════════════════════════════════════════════════
     2. NAVBAR SHADOW ON SCROLL
     Adds .scrolled to .navbar once the user scrolls past 10 px.
     CSS in style.css handles the visual change.
  ════════════════════════════════════════════════════════════════════════ */

  function initNavbarScroll() {
    var navbar = $('.navbar');
    if (!navbar) return;

    function onScroll() {
      if (window.pageYOffset > 10) {
        navbar.classList.add('scrolled');
      } else {
        navbar.classList.remove('scrolled');
      }
    }

    window.addEventListener('scroll', debounce(onScroll, 10), { passive: true });
    onScroll(); // Run once on page load in case the page is already scrolled
  }

  /* ═══════════════════════════════════════════════════════════════════════
     3. ANALYZE BUTTON — LOADING ANIMATION + INPUT VALIDATION
     Validates the form fields before submit, then shows a spinner and
     disables the button so it cannot be double-clicked.
  ════════════════════════════════════════════════════════════════════════ */

  function initAnalyzeButton() {
    var form     = $('#reviewForm');
    var btn      = $('#analyzeBtn');
    if (!form || !btn) return;

    var movieInput  = $('#movie_name');
    var reviewInput = $('#review');
    var btnText     = btn.querySelector('.btn-analyze__text');
    var spinner     = btn.querySelector('.spinner');

    form.addEventListener('submit', function (e) {
      // Clear previous errors first
      clearErrors();

      var valid = validateForm(movieInput, reviewInput);

      if (!valid) {
        e.preventDefault(); // Block submission
        return;
      }

      // ── Show loading state ──
      btn.disabled = true;
      btn.classList.add('loading');

      if (btnText) {
        btnText.innerHTML =
          '<i class="fa-solid fa-brain" aria-hidden="true"></i> Analyzing...';
      }
      if (spinner) spinner.style.display = 'block';
    });

    /* Reset button if user navigates back (bfcache) */
    window.addEventListener('pageshow', function (e) {
      if (e.persisted) resetButton(btn, btnText, spinner);
    });
  }

  /* ── Validation logic ──────────────────────────────────────────────── */
  function validateForm(movieInput, reviewInput) {
    var ok = true;

    // Movie name — optional field but if filled in must be non-empty string
    // (Field is optional per the app spec; only warn if whitespace-only)
    if (movieInput && movieInput.value.trim() === '' &&
        movieInput.hasAttribute('required')) {
      showError(movieInput, 'Movie name cannot be empty.');
      ok = false;
    }

    // Review — required
    if (!reviewInput) return ok;
    var rev = reviewInput.value.trim();

    if (rev === '') {
      showError(reviewInput, 'Please enter a movie review before analyzing.');
      ok = false;
    } else if (rev.length < 10) {
      showError(reviewInput, 'Review must be at least 10 characters long.');
      ok = false;
    }

    return ok;
  }

  /** Attach an error message below the input and add .input-error class */
  function showError(field, msg) {
    field.classList.add('input-error');
    field.setAttribute('aria-invalid', 'true');

    // Avoid duplicate error messages
    var wrap = field.closest('.field');
    if (!wrap) return;
    if (wrap.querySelector('.field-error-msg')) return;

    var err = document.createElement('span');
    err.className   = 'field-error-msg';
    err.setAttribute('role', 'alert');
    err.innerHTML   = '<i class="fa-solid fa-circle-exclamation" aria-hidden="true"></i> ' + msg;
    wrap.appendChild(err);
    field.focus();
  }

  /** Remove all error states from the form */
  function clearErrors() {
    $$('.input-error').forEach(function (el) {
      el.classList.remove('input-error');
      el.removeAttribute('aria-invalid');
    });
    $$('.field-error-msg').forEach(function (el) { el.remove(); });
  }

  /** Reset button to its original state */
  function resetButton(btn, btnText, spinner) {
    btn.disabled = false;
    btn.classList.remove('loading');
    if (btnText) {
      btnText.innerHTML =
        '<i class="fa-solid fa-magnifying-glass-chart" aria-hidden="true"></i> Analyze Sentiment';
    }
    if (spinner) spinner.style.display = 'none';
  }

  /* ── Clear field error on user input ─────────────────────────────── */
  function initFieldClearOnInput() {
    $$('.field input, .field textarea').forEach(function (el) {
      el.addEventListener('input', function () {
        this.classList.remove('input-error');
        this.removeAttribute('aria-invalid');
        var wrap = this.closest('.field');
        if (wrap) {
          var msg = wrap.querySelector('.field-error-msg');
          if (msg) msg.remove();
        }
      });
    });
  }

  /* ═══════════════════════════════════════════════════════════════════════
     4. FADE-IN ANIMATIONS — INTERSECTION OBSERVER
     Elements with class .fade-in become .visible when they enter the
     viewport.  style.css handles the opacity / transform transition.
     Also auto-assigns .fade-in to cards that were not given it in HTML.
  ════════════════════════════════════════════════════════════════════════ */

  function initFadeIn() {
    if (!('IntersectionObserver' in window)) {
      // Fallback: just make everything visible immediately
      $$('.fade-in').forEach(function (el) { el.classList.add('visible'); });
      return;
    }

    // Auto-assign fade-in to common card types that don't already have it
    var autoTargets = [
      '.feature-card',
      '.stat-card',
      '.step-card',
      '.cast-card',
      '.chart-card',
      '.score-tile',
      '.info-row'
    ];
    $$(autoTargets.join(',')).forEach(function (el, i) {
      if (!el.classList.contains('fade-in')) {
        el.classList.add('fade-in');
        // Stagger delay based on position within parent
        var siblings = Array.from(el.parentElement.children);
        var idx      = siblings.indexOf(el);
        el.style.transitionDelay = (idx * 60) + 'ms';
      }
    });

    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
          observer.unobserve(entry.target); // Fire once only
        }
      });
    }, {
      threshold : 0.1,
      rootMargin: '0px 0px -30px 0px'
    });

    $$('.fade-in').forEach(function (el) { observer.observe(el); });
  }

  /* ═══════════════════════════════════════════════════════════════════════
     5. COUNTER ANIMATION — STATISTICS CARDS
     Finds elements with data-count="<number>" and animates from 0.
     Falls back gracefully if IntersectionObserver is unavailable.
  ════════════════════════════════════════════════════════════════════════ */

  function initCounters() {
    /* Auto-detect stat values from .stat-card__value elements */
    $$('.stat-card__value').forEach(function (el) {
      // Skip elements that already have data-count set
      if (el.dataset.count) return;

      // Extract numeric part from the text content
      var raw = el.textContent.trim();
      // Match leading number (possibly float like 90.59)
      var match = raw.match(/^[\d,.]+/);
      if (!match) return;

      var numStr  = match[0].replace(/,/g, '');
      var num     = parseFloat(numStr);
      if (isNaN(num)) return;

      // Store original full text so we can preserve suffix (K, %, etc.)
      el.dataset.count      = num;
      el.dataset.countOrig  = raw;
      // Reset display to 0 until animation fires
      el.dataset.countReady = '1';
    });

    if (!('IntersectionObserver' in window)) {
      runAllCounters();
      return;
    }

    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          animateCounter(entry.target);
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.5 });

    $$('[data-count-ready]').forEach(function (el) { observer.observe(el); });
  }

  function runAllCounters() {
    $$('[data-count-ready]').forEach(animateCounter);
  }

  function animateCounter(el) {
    var end      = parseFloat(el.dataset.count);
    var orig     = el.dataset.countOrig || String(end);
    var duration = 1200; // ms
    var start    = null;

    // Determine decimal places from the original number
    var decMatch = orig.match(/^[\d]+\.(\d+)/);
    var decimals = decMatch ? decMatch[1].length : 0;

    // Detect suffix (everything after the number)
    var suffix = orig.replace(/^[\d,.]+/, '');

    // Detect inner <span class="unit"> and extract its content separately
    var unitSpan = el.querySelector('.unit');
    var unitText = unitSpan ? unitSpan.outerHTML : '';
    if (unitSpan) {
      // Remove the unit span from suffix detection
      suffix = '';
    }

    function step(ts) {
      if (!start) start = ts;
      var progress = Math.min((ts - start) / duration, 1);
      // Ease-out cubic
      var eased = 1 - Math.pow(1 - progress, 3);
      var current = end * eased;

      if (unitSpan) {
        // Preserve the <span class="unit"> markup
        var pre = el.innerHTML.replace(unitSpan.outerHTML, '');
        el.innerHTML = current.toFixed(decimals) + unitText;
      } else {
        el.textContent = current.toFixed(decimals) + suffix;
      }

      if (progress < 1) {
        requestAnimationFrame(step);
      } else {
        // Restore original content exactly
        if (unitSpan) {
          el.innerHTML = end.toFixed(decimals) + unitText;
        } else {
          el.textContent = orig;
        }
      }
    }

    requestAnimationFrame(step);
  }

  /* ═══════════════════════════════════════════════════════════════════════
     6. AUTO-HIDE FLASH MESSAGES
     Flash messages fade out and are removed from DOM after 5 seconds.
     Each message also has a close button handled in base.html inline JS;
     this script is the auto-dismiss fallback.
  ════════════════════════════════════════════════════════════════════════ */

  function initFlashMessages() {
    $$('.flash-msg').forEach(function (el, i) {
      // Stagger auto-dismiss so multiple messages don't all vanish at once
      var delay = 5000 + (i * 600);
      setTimeout(function () {
        dismissFlash(el);
      }, delay);
    });
  }

  function dismissFlash(el) {
    if (!el || !el.parentElement) return;
    el.style.transition = 'opacity 0.45s ease, transform 0.45s ease, max-height 0.45s ease, margin 0.45s ease, padding 0.45s ease';
    el.style.opacity    = '0';
    el.style.transform  = 'translateY(-10px)';
    el.style.maxHeight  = '0';
    el.style.margin     = '0';
    el.style.padding    = '0';
    setTimeout(function () { if (el.parentElement) el.remove(); }, 480);
  }

  /* ═══════════════════════════════════════════════════════════════════════
     7. CHARACTER COUNTER
     Live counter below the review textarea.
  ════════════════════════════════════════════════════════════════════════ */

  function initCharCounter() {
    var textarea = $('#review');
    var counter  = $('#charCount');
    if (!textarea || !counter) return;

    var MAX = parseInt(textarea.getAttribute('maxlength') || '5000', 10);

    function update() {
      var len = textarea.value.length;
      counter.textContent = len.toLocaleString() + ' / ' + MAX.toLocaleString();
      counter.classList.toggle('warn', len > MAX * 0.85);
    }

    textarea.addEventListener('input', update);
    update(); // Initialise on load
  }

  /* ═══════════════════════════════════════════════════════════════════════
     8. BACK-TO-TOP BUTTON
     Creates the button dynamically (no HTML required), appends it to body,
     shows/hides it based on scroll position.
  ════════════════════════════════════════════════════════════════════════ */

  function initBackToTop() {
    // Create the button
    var btn = document.createElement('button');
    btn.id            = 'backToTop';
    btn.setAttribute('aria-label', 'Scroll back to top');
    btn.innerHTML     = '<i class="fa-solid fa-arrow-up" aria-hidden="true"></i>';
    document.body.appendChild(btn);

    var THRESHOLD = 300;

    function onScroll() {
      if (window.pageYOffset > THRESHOLD) {
        btn.classList.add('visible');
      } else {
        btn.classList.remove('visible');
      }
    }

    btn.addEventListener('click', function () {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });

    window.addEventListener('scroll', debounce(onScroll, 80), { passive: true });
    onScroll();
  }

  /* ═══════════════════════════════════════════════════════════════════════
     9. DARK DASHBOARD INTERACTIONS
     Collection of small UX details that elevate the dashboard feel.
  ════════════════════════════════════════════════════════════════════════ */

  function initDashboardInteractions() {

    /* ── Ripple effect on all .btn clicks ─────────────────────────────── */
    $$('.btn, .btn-analyze, .btn-primary, .btn-ghost').forEach(function (btn) {
      btn.addEventListener('click', createRipple);
    });

    function createRipple(e) {
      var el   = this;
      // Skip if button is disabled
      if (el.disabled) return;

      var rect   = el.getBoundingClientRect();
      var size   = Math.max(rect.width, rect.height) * 1.8;
      var x      = e.clientX - rect.left - size / 2;
      var y      = e.clientY - rect.top  - size / 2;

      var ripple = document.createElement('span');
      ripple.style.cssText = [
        'position:absolute',
        'border-radius:50%',
        'pointer-events:none',
        'width:'   + size + 'px',
        'height:'  + size + 'px',
        'left:'    + x    + 'px',
        'top:'     + y    + 'px',
        'background:rgba(255,255,255,0.18)',
        'transform:scale(0)',
        'animation:rippleAnim 0.55s ease forwards'
      ].join(';');

      // Inject keyframes once
      if (!document.getElementById('rippleStyle')) {
        var s = document.createElement('style');
        s.id  = 'rippleStyle';
        s.textContent = '@keyframes rippleAnim{to{transform:scale(1);opacity:0}}';
        document.head.appendChild(s);
      }

      el.appendChild(ripple);
      setTimeout(function () { ripple.remove(); }, 600);
    }

    /* ── Card tilt effect on mouse move (desktop only) ────────────────── */
    if (window.matchMedia('(hover: hover)').matches) {
      $$('.glass-card, .feature-card, .step-card, .stat-card').forEach(function (card) {
        card.addEventListener('mousemove', function (e) {
          var rect   = this.getBoundingClientRect();
          var cx     = rect.left + rect.width  / 2;
          var cy     = rect.top  + rect.height / 2;
          var dx     = (e.clientX - cx) / (rect.width  / 2);
          var dy     = (e.clientY - cy) / (rect.height / 2);
          var tiltX  = (dy * -4).toFixed(2); // max ±4°
          var tiltY  = (dx *  4).toFixed(2);
          this.style.transform = 'perspective(700px) rotateX(' + tiltX + 'deg) rotateY(' + tiltY + 'deg) translateY(-4px)';
        });

        card.addEventListener('mouseleave', function () {
          this.style.transform = '';
        });
      });
    }

    /* ── Result page: copy button feedback ───────────────────────────── */
    var copyBtn = $('#copyBtn');
    if (copyBtn) {
      // The inline handler in result.html handles clipboard API;
      // this adds a CSS transition class for extra polish.
      copyBtn.addEventListener('click', function () {
        this.classList.add('btn-copied');
        setTimeout(function () { copyBtn.classList.remove('btn-copied'); }, 2100);
      });
    }

    /* ── Auto-focus review textarea on hero CTA click ────────────────── */
    var heroCta = $('.hero__cta');
    var reviewTA = $('#review');
    if (heroCta && reviewTA) {
      heroCta.addEventListener('click', function () {
        setTimeout(function () { reviewTA.focus(); }, 500);
      });
    }

    /* ── Input placeholder shimmer on focus ──────────────────────────── */
    $$('.field input, .field textarea').forEach(function (el) {
      el.addEventListener('focus', function () {
        var label = this.closest('.field') && this.closest('.field').querySelector('label');
        if (label) label.style.color = 'var(--sentiment-primary, #e50914)';
      });
      el.addEventListener('blur', function () {
        var label = this.closest('.field') && this.closest('.field').querySelector('label');
        if (label) label.style.color = '';
      });
    });
  }

  /* ═══════════════════════════════════════════════════════════════════════
     10. RESULT PAGE — CONFIDENCE BAR + DONUT RING INIT
     In case result.html's inline script hasn't already run these
     (belt-and-suspenders), we trigger them here too.
  ════════════════════════════════════════════════════════════════════════ */

  function initResultPage() {
    var page = $('#resultPage');
    if (!page) return; // Not on the result page

    var confStr = page.dataset.confidence || '0%';
    var confNum = parseFloat(confStr) || 0;

    /* ── Animate confidence bar ─────────────────────────────────────── */
    var bar = $('#confBar');
    if (bar && bar.style.width === '0%') {
      requestAnimationFrame(function () {
        requestAnimationFrame(function () {
          bar.style.width = confNum + '%';
        });
      });
    }

    /* ── Animate donut ring ─────────────────────────────────────────── */
    var donut      = $('#donutChart');
    var donutLabel = $('#donutLabel');
    if (donut && donutLabel && donutLabel.textContent.trim() === '—') {
      donutLabel.textContent = confNum.toFixed(0) + '%';
      var start  = 0;
      var dur    = 900;
      var startT = null;
      function animDonut(ts) {
        if (!startT) startT = ts;
        var p   = Math.min((ts - startT) / dur, 1);
        var val = start + (confNum - start) * (1 - Math.pow(1 - p, 3));
        donut.style.setProperty('--p', val.toFixed(1));
        donutLabel.textContent = val.toFixed(0) + '%';
        if (p < 1) requestAnimationFrame(animDonut);
      }
      requestAnimationFrame(animDonut);
    }
  }

  /* ═══════════════════════════════════════════════════════════════════════
     11. MOBILE NAV — keyboard trap + Escape to close
  ════════════════════════════════════════════════════════════════════════ */

  function initMobileNav() {
    var toggle   = $('#navToggle');
    var navLinks = $('#navLinks');
    if (!toggle || !navLinks) return;

    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && navLinks.classList.contains('open')) {
        navLinks.classList.remove('open');
        toggle.setAttribute('aria-expanded', 'false');
        toggle.focus();
      }
    });
  }

  /* ═══════════════════════════════════════════════════════════════════════
     INIT — boot all modules when DOM is ready
  ════════════════════════════════════════════════════════════════════════ */

  function init() {
    initSmoothScroll();
    initNavbarScroll();
    initAnalyzeButton();
    initFieldClearOnInput();
    initFadeIn();
    initCounters();
    initFlashMessages();
    initCharCounter();
    initBackToTop();
    initDashboardInteractions();
    initResultPage();
    initMobileNav();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    // DOMContentLoaded already fired (script loaded with defer)
    init();
  }

}());
