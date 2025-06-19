# 5/27/24

- A powerful technique for navigating complex problem domains: adjacency maps.

  - In these kinds of domains, it’s unclear where to get started.

    - There are hundreds of possible toeholds, in a big swirling chaotic soup of possibility.

      - It’s unclear which toeholds will turn out to be viable.

      - A number of toeholds have obvious adjacencies to grow into if they turn out to be viable.

      - Which toehold to start with is not just its own value, but the indirect value of the adjacent things to grow into.

    - It’s hard to decide which toehold to start at if you’re just one person.

    - But *coordinating* that choice is even harder with a team.

      - The difficulty of coordinating the choice goes up super-linearly with the size of the team.

  - That’s where adjacency maps are useful.

    - Think of them like a tech tree from the game Civilization.

    - First, you ask everyone in the team to come up with every toehold of value that *might* plausibly have even one user who would find it valuable.

      - You want a comprehensive enumeration of all possible use cases.

      - The bar for inclusion is deliberately very low, favoring comprehensiveness.

    - Then, you factor them into a tree.

      - Each node represents a use case.

      - Use cases have a connection to a parent use case if they are dependencies of the parent.

        - That is, to build the use case you would have had to at least build out the parent use cases and then layer on more work on top.

        - The dimensions that they can be dependencies:

          - Engineering - Code that has to be written

          - UX - User experiences that have to be designed

          - Data - Any data quality that has to be above some quality threshold.

          - Behavior - Any user behaviors and expectations that have to be established to activate this use case.

        - For each dimension, “t-shirt size” the expected effort.

      - Lay out the tree so that use cases expand from left to right, with the root note at the left.

  - Having this shared adjacency map changes the game.

    - Now everyone can see that their use case is on the map.

      - They don’t have to continually advocate for it, because they can see it’s on the map.

    - People can see that items on the left have fewer dependencies than those on the right.

    - People can also see which use cases are “upstream” of nearly everything.

      - “We all see that this use case exists. We disagree on how valuable it is, but we all agree that it is upstream of almost every use case. So we can agree to prototype it.”

    - When you prototype a use case, if it’s getting good traction, invest more into it, up a ladder of audiences:

      - The person building it finds it surprisingly compelling

      - The team finds it surprisingly compelling

      - A broader internal dogfood group finds it surprisingly compelling

      - An external trusted testers group finds it surprisingly compelling

      - An external beta finds it surprisingly compelling

      - General release

    - At each time step, pick the use case with the fewest unbuilt dependencies.

      - If any audience found it surprisingly compelling, invest a bit more to get to the next rung of the audience ladder.

      - At any point if interest wanes from the expanded audience, pick the next use case that is downstream of this use case that has the most value downstream of it.

    - As you go and learn more, update your map with more use cases and better estimates of costs.

  - The typical approach to this kind of problem is to have a charismatic and powerful lead pick a single vision.

    - This solves the coordination problem, but at the cost of being *significantly* less resilient.

    - The vision is likely to be wrong in such a complex and unknown environment.

    - If the vision the leader picked was wrong, the project will fail.

    - Even if the leader projects total confidence, the organization will likely gossip and whisper about oversights in the plan, undermining the likelihood it actually works.

    - The leader’s confidence is fundamentally a show, bravado, unearned confidence, designed to be strong enough to coordinate the team to.

  - The adjacency map process is wildly different from the typical approach; it resolves a coordination challenge *and* allows adaptation and growth as you surf through the problem space.

  - Instead of pretending you know everything and have an infallible vision, you embrace the uncertainty, and surf it.

  - Of course, this is a thing that I built a hobby open-source project around a couple years ago: [<u>https://github.com/jkomoros/adjacency-map</u>](https://github.com/jkomoros/adjacency-map)

    - The project allows creating a shared interactive map, and also comparing different “scenarios” (e.g. different costs and value guesses from different parts of the team) to navigate the map.

- Meta-ecosystems are powerful forces.

  - There are ecosystems (a single platform) and meta-ecosystems (multiple platforms that are all affiliated and act, in some ways, like one).

  - The difference between them is how many different entities are making coherent decisions about what behavior to ship.

  - There are two dimensions: the specification of behavior and the implementation.

  - Proprietary platforms are almost always shipped by one entity, and the "specification" is simply the behavior.

  - Even open platforms that have only one implementation mostly act like one platform, because although people could tweak the implementation they ship, it's not worth the effort and everyone generally ships the single implementation.

  - The web platform is an open specification and a handful of open implementations, a meta ecosystem.

  - Email is also a meta-ecosystem; lots of different implementations, and also a kind of fuzzy set of standards.

  - An open system's canonical standards are not some crisp, clear single entity; it's a fuzzy superposition.

    - At the core, the subset of things that most people agree are part of the specification.

    - The fuzzy edges are the parts that only a subset of people believe is canonical.

    - You could say the specification is centralized if observers expect or believe that the spec is basically one (possibly fuzzy) thing, as opposed to multiple things.

  - Ecosystems have power that scales with the square of the number of participants.

    - So two ecosystems that have the same notional specification, join together into a meta ecosystem, and get significantly more powerful than either entity could do alone.

  - They are joined together into one ecosystem by an elastic band.

    - The elastic band is the expectation and belief that they represent one, fuzzy, thing.

    - If it were an iron band, and they had to move forward in lockstep, then the slowest implementation could slow the whole ecosystem down for everyone.

    - As an elastic band, different implementations can go in different directions they think are good, initially stretching the band and sticking their necks out.

    - But if other implementers and users agree it's a good direction, they'll also go in that direction, putting increasing pressure on the laggards.

    - If any entity pulls too far in any direction, the band breaks and they are no longer part of the meta-ecosystem, a discontinuous loss of ecosystem energy for them.

      - Because of that discontinuous loss of energy, most entities will avoid doing that.

      - Only if an entity represents the vast majority of the ecosystem individually might going it alone work, but it would be an aggressive and bad-faith act.

      - “Embrace, extend, extinguish”

    - In this way the fuzzy ecosystem is constantly evolving, and moving in a coherent, consensus direction with a collective, ongoing vote on legitimacy of a roiling chaos of micro-decisions.

    - It's meta-stable.

  - When do things pull into these meta-ecosystems vs stay separate as proprietary ecosystems?

    - The first thing that is necessary is that a critical mass of participants think of them as a thing with a (perhaps fuzzy) spec.

    - That's possible if enough ethos aligns... but also critically if the holders of the IP also are OK with joining.

      - You can't have an adverserially-meta ecosystem emerge out of a proprietary platform unless the IP holder releases their rights or never had them in the first place.

    - But once you have that, it can emerge if things are *mostly* coherent and the same.

    - This can only happen if things are similar enough, and also changing slowly enough to mostly cohere.

    - This is only really possible in late stage ecosystems where a stable, viable thing has already sublimated out of the chaos.

    - If you try to do this at too high of a pace layer, the overall meta distillation cannot cohere because the individual components have too much chaotic energy.

    - Once a meta ecosystem coheres out of the chaos, it tends to get stronger and stronger (and slower and slower) and the longer the precedent for it existing, the less likely it is to decohere in the future.

- Things have *the quality that cannot be named* if they were evolved from real use, not designed.

  - *The quality that cannot be named* is a concept from Christopher Alexander, for things that have a resonant wholeness to them.

    - You might call this sublime beauty.

  - This process is kind of like stable diffusion.

    - You take effectively random noise to start.

    - If it’s viable, then you continuously evolve it.

    - You evolve and improve the parts that are most important (or most broken).

    - Over time it evolves from a random mess into something sublimely beautiful.

  - You can sense the souls of the swarm of users in the past who improved it and shaped it.

    - This gives the thing life and vitality.

    - Not *in theory* beautiful, *in practice* beautiful.

    - The beauty of viability and adaptability.

  - This process works extremely well if the curators have calibrated taste.

    - The curators can sculpt the random, messy underlying thing into a more beautiful version of itself.

    - The curator needs real-world, relevant experience to craft their taste in that domain.

    - A gardener will have better taste in gardening than a chef will in gardening.

    - Ecosystems that can maximize the indirect effects of the taste of their best users can create a ton of value.

  - Javascript is sublimely beautiful.

    - It was created in just 10 days!

    - The creator had experience in designing languages, but was given a last-minute constraint to make it look like Java.

    - The language was rough, and a bit odd… but it worked, and was affixed to an open, ubiquitous system, so it got tons of use.

    - And now Javascript (especially the Typescript variant) have evolved to be, if not a perfect language, at least a great one!

    - Sublimely beautiful!

- Expectations are high when the creator says "I think this is a good thing and you should use it."

  - Users blame the creator (rightfully) if they don't end up liking it.

  - But when the creator says "I think this thing sucks but if you want to use it feel free to crawl through this broken glass" and it sucks the user can't blame the creator.

  - That makes the thing more resilient to the downside risk of users having such a bad time that they never use it again.

  - Self-capping downside.

- Illegible + open gives you capped downside while retaining exposure to upside.

  - Many times you don’t know if a given feature that’s being built is viable or not.

  - One way to do that is to build it carefully in a cave until it’s of very high quality and then release it.

    - But this takes *tons* of time… and you still might be wrong.

    - High quality on a useless thing is useless.

  - Another approach is to iterate in public.

  - The metric to optimize is “Minimize the number of people who use it and have such a bad experience they never want to use it again, while secondarily maximizing usage.”

    - One way to clear this bar is to have a perfect feature.

    - Another way to clear this bar is to have a resilient / forgiving audience.

  - You can get a resilient audience by having them self-select through a “gauntlet” that only the most motivated users make it through.

  - Openness means that people can self-select into your system and you don’t need to find them.

    - You might be surprised where your best users come from.

  - Illegibility creates a kind of gauntlet.

    - Low-motivation people who happen across the feature will think “... what is this?” and bounce.

    - Only motivated users will stick with it to figure out what the thing is.

    - Motivated users are inherently more resilient.

      - If the feature isn’t valuable, they think to themselves “well they did warn me…”

    - A few ways to create illegibility:

      - Don’t have much polish (leave typos, etc).

      - Don’t have clear explainers or READMEs.

      - Have documentation in a Google Doc, not a website.

      - Assume background context the reader might not have.

      - Connect nine of the ten dots.

      - Hide signal in a swarm of interesting but distracting details.

- At the beginning, do the simplest, most constrained thing that might work.

  - Then gradually relax the constraints when you run into problems.

  - Conceptually related to the [<u>Rule of least power</u>](https://www.w3.org/2001/tag/doc/leastPower.html)

    - Or the IETF’s rough consensus and working code.

  - This works because you cut down to the very core, which means you don't have the ballooning complexity from accommodating edge cases, you simply cut them out of the possibility space.

    - You get less possibility space, but in a trade for them being significantly more likely to be viable and self-consistent.

    - Then you can grow from viability into more capable over time through adaptation.

- In design, a probe is a thing to catalyze feedback, a concrete thing to react to.

  - A probe is not designed to be correct.

  - It's designed to attract feedback.

  - Pick a maximally interesting "what if" question whose answer will be maximally clarifying and then trying to make it concrete to interact with.

- A magic trick for keeping things feeling organic: have a secret plan that you don't share.

  - When something happens according to plan, the vibe is "What a crazy random happenstance!"

  - If it goes against plan, you don't lose face or apparent momentum, because as far as everyone else knows there is no plan.

    - Self-capping downside.

  - If it goes to plan, then people think you got really lucky... which is fine if you don't need to get credit for causing it.

  - You also can smoothly adapt and change your plan as you learn more.

  - This is part of the Radagast magic.

- Search is your guide through an open-ended jungle of the web ecosystem.

  - Everyone benefits.

  - The jungle is open-ended and permissionless, so everyone in the jungle benefits.

    - No one can tell any part of the jungle they can't exist.

  - The user can choose whichever guide they want.

    - Or choose to go it alone!

    - The user benefits significantly by having a calibrated guide.

  - The guides can compete on the quality of their guidance.

    - If their guidance gets better as more users use them, they get a more defensible quality against their competitors, a benefit that compounds.

  - Open-endedness is beautiful possibility, but also a bit scary and overwhelming.

  - This combination between an open ecosystem and a high-quality, optional guide is a powerful complementarity.

- For things that are privacy sensitive, the epsilon matters a lot.

  - Epsilon is a concept from differential privacy.

  - The epsilon says how likely two distinct datasets that differ by one record could be distinguished (revealing the presence or absence of that record).

    - A smaller epsilon is more private.

    - If the epsilon is sufficiently tight, your data is hiding in a tornado of other data, and its presence can't trace back to you.

    - That said, low epsilons typically make it harder to achieve the same amount of utility in the systems. It’s a tradeoff.

  - How much a given feature might interfere with our privacy comes down to the epsilon.

    - There can be different epsilons for *others* in the ecosystem (what data might an outside observer be able to figure out about you) and also for the provider (what data might the provider themselves be able to detect from their privileged access to the data set).

    - In the default architecture of our current paradigm, the service provider has full, unfettered access to the totality of their data, but still might not be *doing* anything with some data, and so crossing the line to sift through it, even in an automated way, may change people’s perceptions.

  - A "Do you want to opt out of this chat service training on your DMs" comes with an implied "... at what epsilon?”

  - The “at what epsilon” is rarely addressed.

  - The epsilon matters fundamentally… and yet high- and low-epsilon implementations are presented to users the same way.

  - A bad high-epsilon feature can taint the user’s perception of all such features, even ones that would be implemented in a low-epsilon way.

- The 4-up evolution works great for images, less great for text.

  - Branching text interfaces are less satisfying than branching image generation.

  - Because text has to be serially scanned to be absorbed (whether being read or being heard).

  - But images are absorbed as a field, all at once.

    - You can perceive the image as a whole, including in your peripheral vision, allowing your eyes to dart to any part of the image that it wants more detail on without delay.

    - Wittgenstein pointed this out in his Picture Theory of Language.

- A chat loop is an error correcting loop that doesn't feel like an error correction loop as a UX.

  - Ten blue links is also an error-correcting loop, it allows some slop.

  - "I'm feeling lucky" is a "yup this definitely works", a very high quality bar to clear, because there is no error correcting loop (except hitting the back button).

- A prompting hack: after each message from the LLM reply "Did you make a mistake in your last message?".

  - LLMs are better at analyzing their response after the fact than when it's unspooling its answer.

  - This is true even if it’s the same LLM that generated the text.

  - When it's generating text, it’s YOLOing it token-by-token, it can accidentally back itself into a corner that it can't correct.

  - But after the fact it can say "yeah, that got stuck in a corner, here's the way out of it" and do a better job the next time.

- Comments on fields in an API are now more important.

  - It used to be that computers could only understand the formal types, and the comments were only useful for humans reading the code.

  - … But LLMs might read and write code now too, and the comments and metadata about a field or property give them additional fuzzy semantics to work with!

- LLMs are not open-ended.

  - (At least in current architectures)

  - They are crystallized at a moment in time; after they are trained, they do not change or adapt.

  - If you want an open-ended system, it can have an LLM as a component, but it can’t *be* the LLM.

- LLMs allow a new lower-effort tier of prototyping.

  - Before, the way to prototype was just to do it rough and ready manually, with throw-away code.

  - But now there's an easier / cheaper tier before that point: simply ask an LLM to do it.

  - Use LLMs to duct tape something together, cache the result, and then later replace it with traditional code if you want to keep it and use it a lot.

  - In my [<u>Code Sprouts hobby project</u>](https://github.com/jkomoros/code-sprouts), I needed a blank javascript object that matched a given typescript schema.

    - I could have theoretically found a Typescript parser that could run in a browser and then do some kind of processing on the AST to generate an object.

    - But that’s a lot of work and it’s a simple hobby project.

    - Instead, I just asked an LLM to do it, and then cached the answer in a “compiled” version of a sprout.

- A sweet spot for LLMs: conceptually easy, syntactically hard.

- Exaptation: use an existing thing for a novel, unexpected purpose.

- It’s possible to use web tech to build a website that is totally unlike a normal website.

  - Normal websites assume a centralized, canonical, opinionated server.

  - But that part isn't actually required by the model, it's just the convention.

  - You can use the tech and drop the convention.

  - You'll be going against the grain, but you can make lateral things out of it!

- Secure enclaves on phones went from a "no one wants this" to "I can't imagine not having this" over a multi year evolutionary period.

  - A weak gradient but a self-accelerating one.

  - Once it existed, it made sense to move sensitive workloads to it, which then made sense to improve the integrity of the boundary and the capabilities, which then pulled in more use cases...

  - Confidential computing isn't useful for today's mainstream cloud services and architectures.

    - "Of *course* Google can see my gmail data, how could they not?"

    - The current default service architecture *presumes* the service can see all of the data.

  - Confidential computing for end users is more about protecting an application's data from the host cloud... a more esoteric and less obvious need.

  - But confidential computing exists because various high-sensitivity workloads need that protection from the cloud host.

  - You can surf that motivating need for confidential computing to exist to enable something fundamentally new that isn't possible without it.

  - It's now *possible* to build a new type of user-facing, more private software because of confidential computing, even if that's not what it was originally built for and no users knew to ask for it.

- Uncertainty is stressful.

  - But uncertainty is also the source of possibility, the (relative) lack of constraints.

  - If you know what's going to happen already, nothing interesting or surprising can happen.

  - Interestingness is uncertainty collapsing into certainty, the edge of possibility and reality zippering together, the wave function collapsing.

- A scenario: a wide-eyed person goes out into the forest on a quest.

  - "I found this cool plant! Except it had scales. And it breathed fire. And instead of me eating it, it was eating me!"

  - "... That's a dragon!"

- It's scary to put all your data into an app that you just met!

  - The longer you "know" an app, the less scary it is to give it data.

  - You have more time with it to develop trust with it.

- A system that has to coordinate within itself will be slow to build new use cases.

  - It has to coordinate the components to a top-down conception of what should be done.

  - A swarm doesn't have a coherent vision of itself to coordinate to.

    - All of the coordination cost evaporates away.

    - For large systems the coordination cost is the vast majority of the cost to build something.

- Swarms are powerfully asymmetric, but they're hard to interact with.

  - There's no coherent face to talk to, no one neck to choke.

  - Without a neck to choke, it can be hard to trust a swarm.

    - Who is it that you trust?

    - What do you do if the system betrays your trust?

  - Our current privacy model also assumes that all of your data goes to one entity who has a lot of power over that data… which isn’t viable with an anonymous swarm of untrusted entities.

  - But if you can make it so there's one, trusted, consistent interface that synthesizes the results of the swarm, and makes sure the swarm doesn't individually get data about you, you get the best of both worlds.

  - The benefit of a single point of contact you trust, but without the downside of a single coordinating entity.

  - Swarm intelligence, safely.

  - An open aggregator.

- If a wave exists, don't get capsized by it, ride it.

- The only hope of surviving the ground truth is to have continual exposure to it.

- A meme is a virus; transmitted even unwittingly.

  - All a virus “knows” how to do is spread, which is *a*moral.

  - Whether its spreading is good or bad comes down to the payload.

  - Does the payload make things better, is it something people would be OK with if given a choice?

  - In which case it's fine, good even.

  - If it's something that people would have chosen, after the fact, to not receive, then it's bad.

- The most productive working relationships are challenging.

  - They’re tough! That's what makes them so productive!

  - The right dynamic tension is the source of light and insight.

  - The person who complements your blindspots the best will be the most frustrating to work with for you.

    - They’ll keep on pointing out things you missed… and be right!

    - And when they do it, you’ll start off thinking they’re wrong (it’s in your blindspot) and only after a lot of challenging back and forth will you realize they’re right.

  - A goal for a relationship: to be as challenging as possible to create value and insight, while minimizing the chance of ever getting to the breaking point.

- An organization without outlaws and rebels will become brittle and die.

  - A system requires chaos and novel distinctive things within it in order to be antifragile.

- Creation and safety are in tension.

  - Fashions change.

  - To stand out requires you to be distinctive, but that exposes you to being selected against when the fashion changes.

  - How can you make sure that when the fashions change you aren't selected against?

  - The safest strategies for mere survival are to be as bland as possible.

  - But interesting things can only happen from distinctiveness.

  - To create something distinctive requires you to stick your neck out.

  - To be an outlaw, a rebel.

- To make meaning requires taking a perspective.

  - A "fractal view from everywhere" is a view from nowhere.

  - A view exclusively from the balcony can’t tell you what it feels like to be on the dancefloor.

- The curse of knowledge.

  - Knowledge is power.

  - Power we take for granted.

  - What's obvious to you is not necessarily obvious to everyone else!

- The hard part of finding a distilled insight is the finding of it.

  - A distilled insight is a short package of information that leads to an “aha” moment in some of the people who receive it.

  - Those aha moments are the edge of understanding, and if one person has that confusion, others likely do too.

  - Once you have it, the sharing is easy and effective.

  - If the receiver of it doesn't have an aha moment from it, then it only took them a few seconds of opportunity cost to tell them.

  - But if it does have the aha moment, it saves them immeasurable time.

  - When navigating an organization through a complex topic, when a 1:1 has an 'aha' moment, share that with the group.

  - Continue sharing it to larger and larger audiences until you don’t see any new receivers having an “aha!” moment.

- People get to choose if they apply discretionary effort in a project.

  - They can’t be compelled.

  - It’s like volunteer labor.

  - So if they disagree with you, and you're the beneficiary of the volunteer labor, don’t focus on if you’re right or wrong, focus on how *they* feel.

  - Because if they feel that you’re wrong or not listening to how they’re right they will stop giving discretionary effort.

- Ideas have to make it out of your head to be good.

  - Sometimes people have an extremely deep and intricate and fully formed idea in their head... but communicating it is extremely hard.

  - Successful communication is the biggest constraint, and frustrating, and hard, and never perfect.

  - If you say "If everyone already understood what was in my brain, this would be easy.", it’s effectively equivalent to "This would be easy if communication was perfect"

    - This is a smuggled infinity.

  - Communication is fundamentally imperfect and lossy and squishy and frustrating.

    - This is the curse of knowhow.

- Someone who is powerful but bad at communicating will think people around them are dumb.

  - “Why does no one understand this obviously great idea?”

  - Someone who is bad at communicating but not powerful will just have no one want to work with them on the idea.

- Most people are more alike than they realize.

  - When everything is going great, the alike things kind of fade out of our awareness as unimportant.

    - Daniel Gilbert describes our brains as “percentage-of-change detectors”.

  - But then those small differences tend to balloon and self-segregate into deepening fault lines of competitive world views.

    - Like the Robbers Cave Experiment, where kids were split into two groups randomly and then a bitter, self-escalating rivalry emerged.

    - Even in non-charged contexts, our minor differences can escalate into significant conflicts.

    - This happens more easily if everything starts out calm and undangerous.

  - In environments where things are existentially scary and we're united in a common enemy, we can see how we're actually more alike than different.

    - Especially when everyone is more alike to each other than to the enemy.

    - In those cases, we focus on how we’re the same, not how we’re different.

    - Unfortunately, this works based on an us-vs-them dynamic (that is at least turned outward).

    - But us-vs-them dynamics are fundamentally toxic. They show up organically in nearly every context.

- People who are successful often don't understand what made them successful.

  - The individual does things for idiosyncratic reasons, but the environment selects for strategies that are viable in practice.

    - There isn’t necessarily a clear, easy-to-understand or narrativizable *reason* for what ends up being viable.

  - The result is that you get lots of successful people who have no idea what made them successful.

  - The thing they think they're doing and what's *actually* load bearing is likely different.

- If you have to read the employment contract extremely closely for gotchas, then you shouldn't work there!

  - An employer being clever by extracting more in a zero-sum game commits them to adversarial negotiations in future iterations of the game.

  - If you're playing “clever” games, that will eventually come out and become known by your counterparties.

  - Then people will interpret every bit of ambiguity as "how are they trying to screw me?"

  - If you work for a Machiavellian then you have to assume that every leaf that falls is part of a plot or scheme.

- Which tasks can be handed off to E.g. Fiverr?

  - Ones that are cheaper to describe in full fidelity with no ambiguity than to execute.

  - Some tasks are cheaper to execute than to describe with enough specificity for someone else to execute.

  - The Coasian theory of the firm, but for individual tasks.

- Different activities have different learning rates.

  - 1x: Book learning

  - 10x: Indirect experience

    - Games, mentoring, case method

  - 100x: Direct experience

  - It’s possible to abduct direct experience into generalized knowhow.

    - It’s most effective if you use indirect experience to help test and disconfirm the limits of your direct experience.

    - If you do this, you can lever your experience into novel situations safely and cheaply.

- If you steamroll through a context you don’t learn from it.

  - If you steamroll through a thing, you overpower the constraints and don’t learn.

  - It’s when you flutter through it, dancing in the eddies, that you absorb the constraints and knowhow.

  - Learning is hard, it’s being buffeted around, caught up in disconfirming evidence that you can’t ignore.

    - It’s frustrating, but necessary for growth.

- It's hard to sit still and have full-formed insights delivered to you.

  - For example, receiving a lecture from your professor, while you sit there, still, and just absorb.

    - You are fully passive, just a receiver of a pre-formed thing, created by someone else.

  - It can work, if:

    - it's a topic that the receiver is intrinsically interested in.

      - Not that they are obligated to care about

    - The receiver respects their teacher unconditionally.

      - They think they have something valuable they should want to hear, even if they don’t understand it at first.

  - But if either of those are even a little not true, it can be excruciating

    - Like having someone berate you when you're stuck in traffic.

  - Ideally instead you get to be doing it yourself, feeling the wind in your hair, developing your own knowhow, and then the teacher is seasoning your knowhow, improving it, redirecting it a bit.

    - In that case you are co-creating the knowhow, you have some ownership of it, and you can feel the teacher improving it.

  - Even if the teacher is wildly ahead of the student, sometimes the learning gets in a student’s head better if the student is allowed to move under their own power.

  - It feels slower in the moment, but only if you assumed a student that was perfectly motivated and perfectly deferential, which rarely happens.

- Saying “I have no agency” is a defense technique to say “I have no blame”.

  - If you don’t have agency then you can’t take blame.

  - This is the logic behind a victim mentality.

  - The victim mentality says, at the limit, “I have no agency and my oppressor is omnipotent”.

  - How much of a victim you are is a spectrum.

    - It’s never 0% or 100%.

    - In some extremely toxic situations it really is near 100%.

    - But in more typical day-to-day situations, often if we don't like the way it's going, we assume it's more than it actually is.

  - A victim mentality tends to fester and self-accelerate.

    - You tell yourself you don't have agency so you don't take the agency, so you have worse outcomes which you resent more, making you even more of an aggrieved victim.

- People are less likely to flip a table that they helped set.