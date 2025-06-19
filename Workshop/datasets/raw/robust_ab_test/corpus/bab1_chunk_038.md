# 3/11/24

- Today I wanted to share with you the deepest magic I know, a kind of alchemy: the <span id="s0cfteif5ebc" class="anchor"></span>nerd club.

  - You might also think of these as a secret garden on a rooftop.

    - Our day jobs are like the hustle and bustle of the city: urgent, finite.

    - Moments of shared reflection are like wandering in a garden with loved ones: important, infinite.

    - The magic is to create a secret garden on a rooftop, surrounded by the hustle and bustle of the city, but outside of it.

  - Steps to create it:

    - 1\) Create a secret, optional group called something like "Navel Gazers."

      - When people hear the name they should say "That sounds like a lame club for nerds."

      - To which you respond, "Yup! …Do you want in?"

      - This makes sure that the only people who join are intrinsically motivated to participate.

    - 2\) In the group, set a fundamental "Yes, and" norm.

      - It's optional and secret, so none of the discussions matter for the surrounding context.

      - If someone says something you think is uninteresting and not worth exploring, you are free to not engage.

      - But if you do engage, you should build on the idea.

      - If you want to engage with something you don't like, you should say something like "Oh, that's interesting! For these kinds of problems I often apply the FOO lens. I wonder if that applies here?"

      - That makes it about you, not them.

      - That is, not forcing someone to fit their seedling idea into whatever frame you have.

      - This helps keep the risk of sharing new ideas very low, and encourages people to do it.

        - The worst you might get is crickets.

      - This is something that everyone who participates should do.

      - This norm should be maintained and refereed by a very active and visible gardener.

    - 3\) Trickle in a small number of new perspectives, continuously.

      - Aim for perhaps 1 to 3 new perspectives added a week.

      - This helps make sure it doesn't get stale and also the norms don't scramble with an influx of new people.

      - The people you add should:

        - 1\) Be very unlikely to ruin the vibe.

          - It only takes one person to poop a party.

          - This is a bar to satisfice, not maximize.

        - 2\) As different from the other people currently in the group as possible.

          - Have a lot of engineers and a salesperson wants in? Great!

          - Have a lot of senior people and someone junior wants in? Great!

          - This helps give a kind of “novelty search” of perspectives.

          - This is a thing to maximize.

      - Make sure to encourage people to engage, and have signals of fun things happening that keep participants who are paying less attention want to pay more attention.

      - Make sure there’s a bit of indirect offgassing of the group that will allow motivated adjacent people to sense the group and discover it.

  - You'd think that this pattern would result in only frivolous discussions, but it actually does exactly the opposite.

  - The discussions are some of the most profoundly rigorous you can imagine.

  - It's optional so participants only invest in ideas they think are interesting, and because it’s a diversity of perspectives it finds disconfirming evidence (or viral resonance) in ideas quickly.

  - This means that ideas that gather energy in the group are very likely to be viral and game-changing.

    - That is, interesting and novel to a diverse set of people the ideas will collide with in the surrounding context.

  - The result is a self-catalyzing meaning-making machine that participants find valuable for its own sake (an infinite game) that regularly spits out, on a stochastic basis, miraculous, game-changing ideas for the surrounding context.

  - This is the kind of magic that you won’t believe until you’ve seen it (and felt it) yourself, but once you have you won’t be able to forget it.

- Extraction and growth are in tension.

  - If you have a thing that creates secondary benefits, then if it’s growing, let it grow!

  - “How do we monetize this seedling” is the wrong question, especially if the thing is only useful strategically if it grows into an oak tree.

  - The fact it’s sprouting is a miracle; trying to monetize it and over-extract is liable to kill the miracle before it gets strong enough to provide significant indirect value.

  - A annoyingly catchy song from *The Lorax* movie my 4 year old loves and that I’ve been forced to listen to more times than I can count: “Let it grow!”

- Writing hand-rolled SQL to work with potentially-sensitive data is an escape hatch in internal systems.

  - It needs to be possible, of course.

  - But every time it happens it is kind of a bug; it’s dangerous, expensive, error prone.

  - Like any escape hatch, you want to analyze the use cases that require use of the escape hatch and create safer / higher-level alternatives for those use cases.

  - Grow the overall use of the system while minimizing the number of times that people *have* to use the escape hatch.

  - For example, you could make derivative tables that denature the sensitive data (e.g. enforce some threshold of k-anonymity) that most of the internal users use, or create higher-level UIs that don’t allow seeing individual data points.

- [<u>Robustly tolerable beats precariously optimal when the downside risk is high</u>](https://www.askell.blog/when-robustly-tolerable-beats-precariously-optimal/).

  - Robustly tolerable means a thing that is “good enough” in a diversity of realistic scenarios.

    - It is rarely non-viable.

  - Conditions and contexts change more often than we intuitively think they will.

  - This means that optimized things are dangerously brittle.

  - Efficiency is often in tension with resilience.

- Some strategies have a lot of wiggle room.

  - Imagine for example there is a massive secular trend happening that is leading to the growth of a seedling product of yours that has a sustainable revenue model, and that seedling is inherently very sticky.

  - In that case, the LTV of customers is going to be huge; you can give away a lot of freebies to get them onto it before they go to a competitor and get stuck there.

  - You don’t have to worry about getting the math precisely right, because the size of the long-term aggregate LTV is so high that there’s a ton of wiggle room.

  - A robust strategy!

- A platform is a random collection of functionality that dares to dream that it's not just a grab bag, but a Thing.

  - By seeing itself as more than the sum of its parts, it can become more than the sum of its parts.

  - It's only by seeing yourself as an end in and of itself that you can thrive.

- Doing a TAM analysis will undercount the value of a TAM-creating thing.

  - TAM assumes a static world (or at least, one that doesn't react to what you do).

  - But if you're unlocking new potential, then your product is creating value in the world, not just harvesting value.

- Quality has to be your problem, not someone else’s.

  - One way is to have every person responsible for QA and have no specialized QA people.

    - This can work well… but if the norm of quality erodes, you might not notice and significant problems might lurk.

  - A QA engineer embedded as a peer in the team is better than an offshore QA engineer.

    - In the former, they are someone to constantly argue for and make the case for quality.

      - The internal tension in the team is healthy and creates a rigorously high-quality result.

      - It’s impossible for the QA person to forget to optimize for quality when they are busy; it’s their job description!

    - In the latter it's some poor schmuck to clean up your messes and blame.

- One of my favorite design methodologies is lateral thinking with weathered technology.

  - (Note: it’s often mistranslated as *withered* technology)

  - The idea is you use only technologies that have become commodity, standardized, robust, boring.

    - Technologies you can take for granted with very little risk.

    - Multiple providers; low cost.

  - The innovation comes not from the components, but from how the components are duct-taped together into novel combinations.

  - The only risk in the system comes from the duct tape for the combination, which is small and cheap.

  - This allows you to try many different combinations quickly and with low risk.

  - Often you can find game-changing combinations out of totally ordinary components: a kind of alchemy.

- One of my favorite parts of the Harry Potter series is in *The Prisoner of Azkaban.*

  - (Spoiler warning if you haven’t read the book!)

  - Harry sees from across the lake someone create an impressive stag patronus to save the day–the same patronus his late father was known for.

  - Harry later loops through time and relives the moment.

  - He runs to where the patronus came from, hoping that maybe, somehow, his father was the one that made it.

  - He doesn’t see anyone else there. As he sees the previous version of himself across the lake, it dawns on him: it’s *him* who generates the patronus.

  - That inspires him to generate the impressive patronus in the same style as his dad and save the day.

  - Don’t wait for someone else to create the kind of situation or environment you want to be in.

  - Create the kind of environment you want to be in.

  - Be the kind of leader you want to work for.

  - Don’t wait for someone else to save you, save yourself.

- The builder mindset can't create more value than what it puts in.

  - It requires a living thing adjacent to the built thing to create growth and upside.

    - For example, an ecosystem of use, or having word-of-mouth growth from your users.

  - The gardener mindset is about focusing most on the living thing and seeing the built thing as secondary (e.g. a trellis).

  - Both builders and gardeners need to rely on built things and living things.

  - The thing that distinguishes them is: which of the two do you focus on first and foremost: the built thing or the living thing?

- One model for running an organization: a small number of chefs and an army of line cooks.

  - This model can allow more efficiently aligning on a specific, opinionated, high-taste vision.

  - A major downside: if there’s a thing in the collective blindspot of the chefs, the organization cannot possibly address it.

  - Only chefs create unexpected upside; line-cooks can only reach the height of whatever thing they were told to execute.

  - Chefs are somewhat like gardeners, line-cooks are like builders.

- Gardener magic is easy to miss even if you’re looking right at it.

  - Because in any given time step it doesn’t look like anything.

  - It’s a magic that arises from continuous growth and motion and can only be seen over time.

  - And even then, it will look like luck or happenstance.

- Optimism is the life force of an organization.

  - If it goes out, the organization succumbs to apathy and dies.

  - Optimism is what makes an organization strive to be better.

  - However, unchecked optimism is what gives rise to **kayfabe**.

    - Kayfabe is optimism to a grotesque, dangerous, self-protective extreme.

  - Kayfabe expands to take all available energy and space you give it.

  - Runaway kayfabe is like a cancer.

    - It demands all of the energy the host can give it, and it will ultimately kill the host.

  - In such an organization, the members of the organization have to use their considerable talents to prop up the kayfabe, the lie, vs creating real value.

    - In organizations with excessive kayfabe, employees are forced to do a thing that’s not good for them and not good for the company, but everyone is compelled to invest time and energy in.

    - A tragedy!

  - The thing that makes organizations healthy is constantly keeping the kayfabe in check.

  - Positive, optimistic, collaborative... but grounded and realistic.

  - Seeing the path for things to grow into amazing things, but be grounded in the constraints of today.

- Robert Conquest’s Second Law of Politics: “The behavior of an organization can best be predicted by assuming it to be controlled by a secret cabal of its enemies.”

- Mihaly Csikszentmihalyi's concept of Flow:

  - If the challenge is too grand you fall into apathy.

  - If the challenge is too weak you fall into boredom.

  - When it’s just right, you can do magic.

- Don’t spend much time worrying about “did I pick the right acorns”.

  - No one pays attention to the acorns you picked that didn’t grow (they are cheap and small and barely worth noticing).

  - What everyone will notice is the acorns that did grow; the oak trees are impossible to miss.

  - This has a nice self-downside-capping property.

  - If it doesn’t work, no one will notice, and if it does work, everyone will notice.

  - So simply spread a diversity of cheap ones and don’t overthink it.

  - This is similar to my advice to people considering blogging but nervous about making their content good enough: “Good news, no one will read it anyway.”

    - As long as it’s not controversial, the only way someone will bother reading it is if it’s good, which is self-capping downside.

- There’s hustle in terms of creation and hustle in terms of kayfabe.

  - The former is what you find in e.g. hobby projects or 0-1 startups.

    - Jumping in to creatively problem solve and build things.

  - The latter is what you find in massive organizations.

    - Focusing on creating process and appearance of motion.

  - These two types of hustle are very different and orthogonal.

- If someone offers you free candy, make sure they aren't trying to lure you into an unmarked van.

- Don't revert to the mean, divert from it.

  - Lean into what makes you special, not what makes you the same.

  - Spread your wings!

- A destructive strategy:

  - Harming your competitor in a way that harms you too... but differentially less to yourself.

  - Everyone loses, you lose less.

  - A net negative for society!

- A no regrets ethos: take the great opportunities in front of you, and don't take the ones that aren't in front of you.

  - This seems simple, but most people focus on opportunities that actually aren't in front of them, and thus aren't live options.

  - Or they don’t jump on a great opportunity because it feels like luck, or doesn’t fit into their preconceived plan.

  - If an amazing opportunity is in your grasp, grab it!

- Someone asked me about my process of making Bits and Bobs.

  - I take live notes during conversations, of little snippets of assertions, observations, principles.

    - (If you’ve ever been in a small group meeting with me, you’ve seen me do this… it’s a bit unnerving if you don’t know what I’m doing, and seems like I’m distracted)

    - These I send to an inbox for me to process later, and often have typos, missing words, only make sense in context, etc

    - These rough notes will only really make sense to me for the next few days.

  - Every few days, I process each of those snippets into its own private working notes card in The Compendium.

    - At this stage I correct typos and add just a bit more color to make it so the idea would stand on its own (perhaps with some study) to myself in the future.

    - At this stage for some ideas I might add 30% more context or framing as I reflect more on the idea and try to make it more general.

    - At this stage, the ideas are ones that should make sense to me even arbitrarily far into the future.

  - At the end of the week I go through the notes from the last week and collect the ones that still strike my fancy as interesting or potentially valuable; something I want to have access to in the future or want to be able to point others to.

    - Often ideas come up multiple times over the course of a week as I implicitly "workshopped" an idea in different conversations; I paste all of the near duplicates next to each other so I can synthesize the strongest / most compelling formulation.

    - I then comb through the extracted snippets.

    - For each I develop them a bit, adding framing questions, breaking into bullets and sub-bullets, perhaps adding a bit of motivation--work that helps fix them and make them stand on their own to a motivated audience that is not me.

    - At this stage I drop out up to 30% of ideas that no longer strike me as being worth developing.

  - Monday mornings I comb through the Bits and Bobs again, making small tweaks, and moving various bits and bobs up or down in the list to make them hang together more thematically.

  - After I publish them, I import them as final working notes into the Compendium and label them; in the future I might publish them within the Compendium.

  - I find that the act of freezing the ideas in amber, of developing them enough to make them stand on their own, makes the strongest formulation of the thought significantly more stable in my head, and much easier for me to naturally recall in the future.

  - I also randomly read a few working notes each day, which helps jog my memory and allow me to particle-collide old ideas, helping generate something new.

  - Every so often I feel the itch to factor out a specific stand-alone essay that hopefully will stand on its own to a less-motivated audience.

- Everything is frustrating.

  - When it's *novel* frustration it's not as challenging or draining.

  - So switch contexts every so often to swap in a different kind of frustration.

  - One of the reasons as a parent I welcome both the weekend and the beginning of the week day.

- The more that there is a surplus of options to choose between, the more important that taste becomes.

  - Taste is the judgment to decide between alternatives in the firehose of options.

  - DJ Khaled can’t play an instrument, but he has impeccable taste (or so I hear).

  - AI creates a massive firehose of content, a cacophonous background noise.

  - In a cacophony, people will flock to the entities with the best taste.

  - Taste is the final moat.

- Interestingness and positivity are not in tension.

  - People often assume that someone who is widely beloved must be boring.

    - Banal, bland.

  - But that's not true!

  - It's possible to be seen as neutral-to-positive by just about everyone and still be interesting.

    - Dwayne Johnson (the Rock).

    - Dolly Parton.

  - Just because everyone has a positive take on them doesn’t mean they aren’t distinctive and interesting.

- Don’t bother convincing other people in the org that your idea is good.

  - Convincing people is extremely expensive, and often doesn’t work.

  - This is especially true when there’s a cacophony of other people trying to convince people of other things, making it harder for your thing to stand out.

  - Don’t bother!

  - Find the people who are already convinced it’s a good idea and just collaborate with them.

  - Successful ideas are naturally viral; more people will notice the momentum and want to join in.

  - So if the idea works, it will get easier and easier to convince people to join in.

- A person with one foot inside the system and one foot outside the system can do magic.

  - They have leverage to change the system directly, but also a vantage point beyond the short-term horizon of the system.

  - However, that person will look crazy (or even dangerous) to people who are entirely inside the system.

  - If there’s a powerful entity with a hammer inside the system, the person straddling the boundary might be in danger.

  - But if there’s no one with a big hammer in the system, then the person only has to fear people thinking they’re kooky, not wanting to remove them from the system.

  - This is one of the reasons open-ended systems are best for game-changing disruptive innovation.

- Tactics that create indirect value in an organization often can’t be understood by it.

  - An organization will find them “kooky”.

  - In an organization you have to be exceeding expectations at your day job, in a traditional, non-kooky way to survive.

  - If you only do the "kooky" stuff you're liable to get knocked out of the game.

  - Even if you have a miracle, people will assume you just got lucky.

  - As you get more senior and have more of an obvious track record of success, people will assume you know what you're doing, and also be more willing to believe that a lucky break was something you caused in some way.

  - Being a systems thinker early in your career is a curse.

    - Your boss will not understand you and maybe even find you threatening.

    - And you won't have a track record to fall back on.

    - You'll see how everything is more complex and requires gardener style approaches, but you won't have earned the flexibility and credibility to do atypical approaches that anyone else might find kooky.

- An emergent requirement of surviving in any organization: pretending your boss is right.

  - Your boss has an extraordinary amount of leverage and sway in whether you should continue working in that organization.

  - Which means that if you don’t even pretend your boss is right, you’re significantly more likely to be knocked out of the game.

  - This is one of the reasons that kayfabe tends to grow, unchecked, in organizations, until it is dangerous.

- Most organizations spend far too much time seeking a small number of miraculous big wins.

  - But most value in practice is from accumulating no-brainers in a way that creates increasingly more value at an accelerating rate.

  - Managers will always focus their people by default on the big boulders, not the small acorns.

  - This is more true the more insecure the manager is and the higher the pressure to show short-term results.

  - "I don't care if that could grow big with low risk, I need a big win *now* or the game might be over for me!”

- As a human in our day to day personal lives, it’s easy to get into an accelerating loop of resentment.

  - In general, if you’re actively looking for things to resent, you’ll find no shortage.

    - The world is noisy and ambiguous.

    - There’s always a shadow to interpret maliciously.

  - One of the cycles of escalating tensions in human systems: if you think you've been aggrieved you think you're justified in hitting back.

    - But the perception of who has been aggrieved and for what is something where you implicitly think you're special, and so you overweight things that have happened to you and underweight things that happen to others.

    - So the other person sees an escalation, and responds to the over-reaction and that accelerates in a loop.

  - When you hold yourself to an impossible standard (e.g. perfectionism) and then fail, your ego will need to find an external cause to make it so it’s not your fault.

    - That external cause will both justify your failure, and your intense emotions you feel about the failure.

    - “I have found the villain and it’s this other person.”

  - This can create a spiral of self-sabotage.

    - You justify bad behavior by inducing others to do things you’re justified in behaving badly in response to.

    - A toxic spiral that tears down not just yourself, but others around you.

  - The way out of this loop? Having compassion for yourself and others.

    - “I’m not perfect, and I never will be because that’s impossible. But I’m good and I can become better with intentional effort. This is true for everyone, not just me.”

- The tech industry has a default ethos of "go fast and assume we're correct and everyone else is dummies".

  - That leads to a lot of moves that look clever but are actually just reckless, e.g. using leverage.

  - Of course, sometimes it really is right, and that’s where some of the disruptive changes come from.

  - Everyone who discovers leverage thinks they're a genius, and go right on thinking they're a genius until it kills them.

- When you’re tinkering directly with a system, you understand it orders of magnitude better.

  - You’re in a loop, interacting with it, making predictions about how it will respond to things you do.

  - This requires you to develop a kind of “theory of mind” of your “opponent”.

    - You’ll develop a visceral sense of “how it thinks”.

  - It’s an active interaction, a dance, not a passive absorption.

  - Your head is in the game, plugged into what’s happening.

- What do you do when a gold rush is on?

  - A gold rush is like a frenzied lottery.

  - A large number of random entrants will win big; most people will win nothing.

  - For example, most railroads back in the railroad boom never made their money back.

  - A similar dynamic will likely happen with large language models: extremely capital intensive to produce, but very little pricing power to make back the investment directly.

  - Betting on any particular gold mine is a lottery ticket.

  - Two meta moves in a gold rush:

    - 1\) Sell pickaxes

    - 2\) Figure out what people will want once gold is abundant

  - These meta moves put you in a good position no matter who wins the individual lotteries of the gold rush.

- Disruptive technologies undermine a fundamental shared assumption.

  - For example: an implied cost structure that everyone can’t imagine ever being different than the way it is.

  - These kinds of disruptions mean that people’s intuitions will be wildly wrong in the face of disruption.

  - The assumption has always held before, so it never made sense to even consider it might change.

- LLMs are not *the* Thing. They’re a *part* of the Thing.

  - What is the Thing? We don’t know yet!

  - LLMs are magical duct tape that can be used to build the Thing.

  - Chat bots are a very natural medium for LLMs, and humans have been willing to meet them where they are so far.

    - LLMs are good enough at it that the interaction pattern feels less like “interacting with a frustratingly baroque and limited phone tree” of previous chat bots.

  - But it’s still not necessarily the most natural interaction pattern for many types of real work for humans.

  - The “kick off a conversation and then let the LLM drive it wherever it’s going to go” is like a leap of faith in the LLM’s abilities on a given task.

  - Other approaches use LLMs not as an autonomous vehicle that drives itself, but as an engine in a car, driven by someone or something else.

  - We’ll look back on this era and say, “why were we so obsessed with chat bots as the Thing?”

- The language model is a component of an overall system.

  - When people imagine something like AGI, they imagine a specific high-powered computer, running a single, finely tuned model, blinking and beeping in a corner somewhere.

  - An alternate view from complex systems is to view the emergent totality of the surrounding system as where the intelligence lies.

  - Not just one model, but multiple interleaved in an emergent web of interactions and systems with real people, traditional computer systems, etc.

  - In this view, any individual model is merely a component.

  - The overall intelligence emerges from a fabric, more akin to the Technium than an engineered object sitting in a corner.

  - A gardener view vs a builder view.

- If you fail at a moderate-upside, well-trodden path, it’s embarrassing.

  - Because there’s an established, effective playbook.

    - Lower returns, lower risk.

  - In the tech world today this might be something like a YC-funded vertical Saas business.

  - But if you take a swing at something massive that would be amazing if it worked but is unlike other things, then if it fails it’s OK, because it was unlikely to work in the first place.

- The web was the original lateral thinking with weathered technology.

  - Hypertext (including links!) existed in a local-only form.

  - The internet existed, too, in a fledgling state.

  - It was just duct taping the two together, to make something, as Tim Berners-Lee's boss famously wrote in the margin of the initial proposal, that was "Vague but exciting".

  - Totally ordinary components with an outcome like alchemy.

  - A disruptive innovation.

- AI is a confusing catch-all term.

  - One way to think about it:

    - AI 1.0: Linear regressions, random forests, etc.

    - AI 2.0: Deep learning. Supervised learning with bespoke, high-quality training data.

    - AI 3.0: Unsupervised learning. LLMs. Messy, kitchen-sink, highly scaled training data.

  - In any given situation, it’s still possible to make a better-performing AI 2.0 specific model.

  - But the notable thing for LLMs is that they’re reasonably good at just about everything.

  - They are robustly tolerable, not precariously optimal.

  - In the early days of computers there was a debate of if specialized ASICs (which can have orders of magnitude better performance) or generalized chips would win out.

  - The latter won out in all but the most performance sensitive niches; the flexibility was just too important.

    - An ASIC takes the logic into a much lower pace layer that is expensive and slow to change.

    - A general purpose chip allows the logic to change in software, many pace layers higher.

  - LLMs aren’t the best at any task; but they’re good enough at a surprising diversity of them.

- Aggregation is a game everyone is forced to play.

  - If there is an ecological niche for an aggregator, and you don't take it, someone else will take it.

  - It's not that you’re betting on being an aggregator, you just want to inoculate yourself against being aggregated.

- You could argue that OpenAI is a middling success.

  - In some ways, obviously, it’s a *massive* success.

  - But OpenAI sought out to change the dynamic of how technology was built in the industry.

  - Instead, OpenAI, in its breakout success, has been captured by the pre-existing dynamic.

  - OpenAI is simply executing the same default aggregator pattern as everyone else.

- The app model doesn’t allow composability.

  - Apps are islands: little monoliths.

  - It’s an extension of the web’s same-origin model… but without the iframe composition primitive.

  - This is one of the reasons apps have a one-size-fits-all shape.

  - Other kinds of experiences (e.g. web apps) can be composed, but not safely… you either get a hellish privacy experience, or a confusing thicket of impossible-to-answer permission prompts.

- Apple is addicted to the app model.

  - They are in the jealous rent-seeking phase of the innovation cycle.

  - This means that there are disruptive moves that Apple is better situated to do than others that they still won’t do because it would disrupt the app model.

  - This means the monolithic, non-composable app model is ripe for disruption.

- Aggregators are gravity wells.

  - All participants are drawn into them because they are incrementally better, and as more people participate the pull on others gets even stronger.

  - The problem is that aggregators are controlled by one entity with god-like powers in that context.

  - What if it was possible to create an *open* aggregator?

  - Something that everyone was pulled into participating in in a gravity well… but that was not owned by any entity, and instead owned by the participants themselves?

- People have gotten burned by closed systems, so to earn their trust with a new type of thing it will have to be open.

  - People don't want a god AI by a single company.

  - A traditional AI aggregator is thinking too small.

- People think of privacy concerns as a tax.

  - A common pattern today in AI frameworks: come up with a clever agent-based LLM approach with composition and then later figure out how to shoehorn privacy principles onto it, a kind of bummer of an asterisk to the system.

  - An alternative: lead with a new privacy paradigm allowing composition first, and then add LLMs to that.

  - This could create a system where a copernican shift of data provides leverage for whole new types of experiences to be possible that weren’t imaginable before.

  - Privacy not as a bummer afterthought but as a fundamental game-changing *enabler*.

- A desirable property in a system: it’s emergently decentralized.

  - Most systems, with more use, tend to become more centralized.

    - Preferential attachment effects mean that people simply use the thing everyone else uses.

  - What if you could design a system that was the opposite?

  - Decentralization is extremely expensive and makes changes in protocols very hard; doing it too early freezes the system in place.

  - You want a system that is not just decentralize-able, but self-catalyzingly decentralized.

  - The system is always sufficiently decentralized, and the bar for what “sufficient” means keeps ratcheting up as more activity happens in the ecosystem.

- One thing LLMs can help with: making simple judgment calls at the level that any reasonable human could do.

  - "This cake recipe says you should put in five tablespoons of tabasco sauce. Is that reasonable?"

  - "This flight search says I need to give my Social Security Number to do a search. Is that reasonable?"

  - These are the kinds of things that are “well, duh” for any reasonable human… but were very hard to do in a formal, computer-friendly way at scale in the past.

  - But LLMs, as a kind of magical duct tape, allow you to take this baseline for granted.

- It’s now possible to take AI for granted.

  - It’s reasonable to assume that there will be at least a handful of models from different providers with GPT 3.5 level of quality, cheap cost, and that contractually agree to not log queries.

  - Of course, it will likely get radically better from here.

  - This means we can now take magical duct tape for granted.

  - The disruption will come from a post-AI entity that takes AI for granted.