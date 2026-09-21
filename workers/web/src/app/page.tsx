import { SignalPreview } from "@/components/signal-preview";

const github = "https://github.com/joeblau/previral";

function Brand({ small = false }: { small?: boolean }) {
  return (
    <a
      href="#"
      className={`brand ${small ? "brand-small" : ""}`}
      aria-label="Previral home"
    >
      <svg viewBox="0 0 36 32" fill="none" aria-hidden="true">
        <path
          d="M3 22V10m7 18V4m7 21V7m7 13v-8m7 5v-2"
          stroke="currentColor"
          strokeWidth="3"
          strokeLinecap="round"
        />
      </svg>
      <span>
        previral<span className="brand-period">.</span>
      </span>
    </a>
  );
}

function Arrow({ diagonal = false }: { diagonal?: boolean }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true" className="arrow">
      <path
        d={diagonal ? "M6 18 18 6M6 6h12v12" : "M4 12h15m-6-6 6 6-6 6"}
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export default function Home() {
  return (
    <>
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <header className="site-header shell">
        <Brand />
        <nav aria-label="Main navigation">
          <a href="#how-it-works">How it works</a>
          <a href="#the-signals">The signals</a>
          <a href={github} className="source-link">
            GitHub <Arrow diagonal />
          </a>
        </nav>
      </header>

      <main id="main">
        <section className="hero shell" aria-labelledby="hero-title">
          <div className="hero-copy">
            <div className="eyebrow mono">
              <span className="status-dot" /> A NEW LENS ON VIDEO
            </div>
            <h1 id="hero-title">
              There’s more
              <br />
              to every <span className="serif-word">frame.</span>
            </h1>
            <p className="hero-description">
              See the signals beneath the story. Explore how video, sound, and
              language relate to predicted brain activity — right on your Mac.
            </p>
            <div className="hero-actions">
              <a className="button button-primary" href="#get-started">
                Build for macOS <Arrow diagonal />
              </a>
              <a className="text-link" href="#how-it-works">
                Take a closer look <span aria-hidden="true">↓</span>
              </a>
            </div>
            <div className="hero-note mono">
              <svg viewBox="0 0 20 20" fill="none" aria-hidden="true">
                <rect
                  x="3"
                  y="3"
                  width="14"
                  height="11"
                  rx="2"
                  stroke="currentColor"
                />
                <path d="M7 17h6m-3-3v3" stroke="currentColor" />
              </svg>{" "}
              BUILT FOR APPLE SILICON <span>·</span> macOS 26+
            </div>
          </div>
          <SignalPreview />
        </section>

        <div className="foundation shell">
          <span className="mono foundation-label">
            RESEARCH MEETS NATIVE PERFORMANCE
          </span>
          <div>
            <span>
              Meta <strong>TRIBE v2</strong>
            </span>
            <span className="foundation-plus">+</span>
            <span>
              Apple <strong>Core ML</strong>
            </span>
            <span className="foundation-plus">+</span>
            <span>
              Native <strong>SwiftUI</strong>
            </span>
          </div>
        </div>

        <section
          className="how-section shell section-space"
          id="how-it-works"
          aria-labelledby="how-title"
        >
          <div className="section-heading">
            <div>
              <p className="eyebrow mono">01 / A DIFFERENT PERSPECTIVE</p>
              <h2 id="how-title">
                From a moving picture
                <br />
                to a picture of <span className="serif-word">response.</span>
              </h2>
            </div>
            <p>
              Bring your video. Previral connects its sights, sounds, and words
              to a synchronized map of predicted cortical activity.
            </p>
          </div>
          <div className="steps">
            <article>
              <span className="step-number mono">01 — INPUT</span>
              <div className="step-drawing film-drawing" aria-hidden="true">
                <span />
                <span />
                <span />
                <div className="film-play">▶</div>
              </div>
              <h3>Start with a video.</h3>
              <p>
                Open a local video file. Its frames, audio, and speech become
                three complementary views of the same story.
              </p>
            </article>
            <article>
              <span className="step-number mono">02 — ANALYSIS</span>
              <div className="step-drawing analysis-drawing" aria-hidden="true">
                <span />
                <span />
                <span />
                <span />
                <span />
                <span />
                <span />
                <span />
                <span />
                <span />
                <span />
                <span />
                <span />
                <span />
                <span />
                <span />
                <span />
                <span />
                <span />
              </div>
              <h3>Follow the signals.</h3>
              <p>
                TRIBE v2 models run through Core ML on your Mac, connecting
                multimodal features to predicted brain responses.
              </p>
            </article>
            <article>
              <span className="step-number mono">03 — EXPLORATION</span>
              <div className="step-drawing response-drawing" aria-hidden="true">
                <svg viewBox="0 0 280 70">
                  <path d="M0 50h40l10-10 12 15 17-41 15 49 13-29 12 10h27l12-26 15 31 12-12h23l10-11 12 24 13-9h37" />
                </svg>
              </div>
              <h3>Find a new perspective.</h3>
              <p>
                Scrub through your video alongside a cortical map and network
                timelines. See how the response changes over time.
              </p>
            </article>
          </div>
        </section>

        <section
          className="signals-section shell section-space"
          id="the-signals"
          aria-labelledby="signals-title"
        >
          <div className="signals-intro">
            <p className="eyebrow mono">02 / MULTIMODAL BY NATURE</p>
            <h2 id="signals-title">
              One story.
              <br />
              <span className="serif-word">Three signals.</span>
            </h2>
            <p>
              A video is more than its visuals. Explore each input’s
              contribution, then see where their responses overlap.
            </p>
            <span className="signal-key mono">
              <i /> SEPARATE INPUTS. SHARED TIMELINE.
            </span>
          </div>
          <div className="modality-list">
            <article className="modality video">
              <div className="modality-icon" aria-hidden="true">
                <svg viewBox="0 0 24 24" fill="none">
                  <rect x="3" y="5" width="13" height="14" rx="3" />
                  <path d="m16 9 5-3v12l-5-3" />
                </svg>
              </div>
              <div>
                <h3>What you see</h3>
                <p>Visual features, movement, and the changing scene.</p>
              </div>
              <span className="mono">VIDEO</span>
            </article>
            <article className="modality audio">
              <div className="modality-icon" aria-hidden="true">
                <svg viewBox="0 0 24 24">
                  <path d="M4 9v6m4-9v12m4-15v18m4-15v12m4-9v6" />
                </svg>
              </div>
              <div>
                <h3>What you hear</h3>
                <p>Sound, rhythm, and the texture of the audio track.</p>
              </div>
              <span className="mono">AUDIO</span>
            </article>
            <article className="modality text">
              <div className="modality-icon" aria-hidden="true">
                <svg viewBox="0 0 24 24" fill="none">
                  <path d="M5 6h14M12 6v14M8 20h8M5 4v5m14-5v5" />
                </svg>
              </div>
              <div>
                <h3>What’s being said</h3>
                <p>Spoken words and the language that gives them context.</p>
              </div>
              <span className="mono">TEXT</span>
            </article>
          </div>
        </section>

        <section
          className="start-section shell"
          id="get-started"
          aria-labelledby="start-title"
        >
          <div className="start-top">
            <p className="eyebrow mono">03 / MADE FOR THE CURIOUS</p>
            <span className="mono source-badge">
              <i className="status-dot" /> SOURCE AVAILABLE
            </span>
          </div>
          <div className="start-content">
            <div>
              <h2 id="start-title">
                Look a little <span className="serif-word">deeper.</span>
              </h2>
              <p>
                Build Previral from source and explore your own videos.
                <br className="desktop-break" /> Your next perspective is
                already in the frame.
              </p>
              <a
                className="button button-primary"
                href={`${github}/tree/main/apple#readme`}
              >
                Get started on GitHub <Arrow diagonal />
              </a>
            </div>
            <div className="requirements">
              <p className="mono">YOUR MAC. YOUR VIDEO.</p>
              <ul>
                <li>
                  <span>Platform</span>macOS 26+ · Apple silicon
                </li>
                <li>
                  <span>Build tools</span>Xcode + XcodeGen
                </li>
                <li>
                  <span>Models</span>Convert locally with Python
                </li>
              </ul>
              <a href={`${github}/tree/main/apple/Conversion`}>
                Explore the conversion tools <Arrow />
              </a>
            </div>
          </div>
        </section>

        <aside className="research-note shell">
          <span className="mono">A NOTE ON THE SCIENCE</span>
          <p>
            Previral visualizes model predictions, not measured brain activity.
            It is an exploration tool, not a measure of attention, a clinical
            tool, or a guarantee of virality. TRIBE v2 model use is subject to
            its noncommercial license.
          </p>
        </aside>
      </main>

      <footer className="site-footer shell">
        <Brand small />
        <span className="mono">A LITTLE SCIENCE. A NEW PERSPECTIVE.</span>
        <div>
          <a href="https://github.com/facebookresearch/tribev2">
            The research <Arrow diagonal />
          </a>
          <a href={github}>
            Source <Arrow diagonal />
          </a>
          <a href="https://joeblau.com">
            By Joe Blau <Arrow diagonal />
          </a>
        </div>
      </footer>
    </>
  );
}
