# 7/1/24

- LLMs’ superpower is translation, from anything to anything.

  - A babelfish.

  - Any translation task will be absorbed over time by LLMs.

  - Lots of things can be framed as a translation problem.

  - For example, going from a UX mock to a working front end is translation.

- An insightful [<u>tweet</u>](https://mas.to/@carnage4life/112593042823322764) from Dare Obasanjo:

  - "There is a big difference between tech as augmentation versus automation. Augmentation (think Excel and accountants) benefits workers while automation (think traffic lights versus traffic wardens) benefits capital.

  - LLMs are controversial because the tech is best at augmentation but is being sold by lots of vendors as automation."

- Society captures useful ideas as cached answers.

  - If the idea is useful, it’s more likely to be maintained or copied.

    - Or written down or transformed into a more durable form.

  - Ideas that are not useful (for example, are not effective) are not maintained; they evaporate away.

  - If there’s a useful idea for your problem, why come up with something new? Just use the thing that is known to work.

  - The swarm intelligence discovers new useful ideas and caches them.

  - Which means that at any time step, if you random sample ideas, the vast majority are useful ones.

  - LLMs then capture all of those useful ideas that someone bothered to write down, embeds them into a crystallized mirror that can be used to reflect society’s cached answers.

  - The best object to help seed humanity if it needs to be rebuilt would be an LLM in a box.

    - A simple black box with a screen, keyboard, that only needs a power source to run.

- Human creators do a thing not unlike LLM synthesis in creative tasks, but with the addition of taste.

  - Good designers absorb lots of good examples from the web of examples they're exposed to, selecting the subset they like (applying their taste) and then synthesizing that intuition into an answer to a given problem in front of them.

  - Not unlike what LLMs are doing.

  - Though LLMs don't have a taste criteria for what to absorb from.

  - Their sampling criterion is "things that humans decided to reproduce", which is aligned with a kind of wisdom of the crowd's generic taste for usefulness.

  - Things that are thought by other humans to be useful will tend to be kept around and that’s what the LLMs sample from.

- If you're going to be wrong, at least be wrong in an interesting way.

  - If your mistakes are boring, then they imply there’s learning you could have done previously but didn’t.

- LLMs today make boring mistakes because they can’t learn.

  - That is, the model’s weights are fixed at training time.

  - Some of the supporting systems around the model that help form the final, end-user-facing experience can be tweaked at a faster rate (e.g. system prompt, data fed in via RAG), but the actual model itself is updated only every few months given the expense.

  - This means that LLMs can’t learn from mistakes they make today.

  - If an LLM makes a mistake and you point it out, it apologizes… but then never gets better from that interaction.

  - Presumably those kinds of interactions will be used during the training of *future* iterations of the model, but that feedback loop is indirect and very long.

    - Compare to for example some search ranking tweaks that can update nearly instantly.

  - Humans are not like this. They quickly absorb disconfirming evidence from mistakes and learn.

  - If you know the thing you’re interacting with will learn, and learn quickly, you have more incentive to be patient and to try to teach it.

- AI is squishy.

  - It’s hard to use a squishy thing to interact with a hard thing.

    - For example, processes that must be extremely reliable and never fail are hard.

  - This is one of the reasons that unsupervised automation is tough with LLMs.

    - Even if you get it to work well 95% of the time, that last 5% of reliability is extremely hard to achieve.

  - If there's a human in the loop it doesn't have to be 100% reliable, the human can figure out when it works and bridge the gap.

  - If there’s no human in the loop, there’s no way to absorb the failure cases, and the viability of the whole system is reduced.

- Individuals are using LLMs way more effectively than organizations today.

  - Individuals are using LLMs in situated contexts, informally and as augmentation to their work, not automation for it.

  - When an individual uses LLMs, there’s a human in the loop every time.

  - The human in the loop can handle failure cases, and steer it more effectively.

  - When an organization wants to use LLMs it has to have some kind of automated process or structure; hard, inflexible, less forgiving.

  - When individuals use AI in their day to day (e.g. employees or students) it’s likely something they don’t want their boss to know about.

  - This means that most successful AI use today is subterranean.

  - It is illegible to the surrounding organization… and the employee has an incentive to *keep* it illegible.

- LLMs recommending the most popular answers (e.g. “Just use tailwind.css”) will lock them into place for longer and more deeply.

  - Society caches answers already, but now the cache will be locked into place more deeply, and will be harder to change.

  - We now have a summarization mechanism that is very effective and also somewhat slow to change.

  - The most popular answers will get ever-more popular. The less popular answers will have a hard time breaking in. The popular answers will have more staying power.

  - We’ll see more hysteresis in the system.

- Code written in the LLM era will be smaller single files, not separations of concerns. LLMs do better with smaller files with all local context.

  - Code that LLMs write will have this quality, and code that is written with LLMs in mind will also have this quality.

- LLMs are inherently statistical summarizers.

  - Which is why they pull to the centroid.

  - "What is the most average answer, conditioned on the input so far?"

- If you have a butler, that butler better work for you, otherwise it's terrifying.

  - Imagine having a butler, but he’s paid for by your overbearing, nosy mother.

  - Or your employer!

- The same-origin paradigm leads to software that is too chunky.

  - That is, it’s bigger than ideal for a given piece of functionality.

  - Each app has to bundle within it a lot of stuff to make its pocket universe viable.

  - This chunkiness can run amok with things like aggregation, leading to apps that are like gravity wells.

  - This also means that any given conceivable app is less likely to be viable.

    - Any single component that is not viable makes the whole assemblage not viable.

    - Apps that are larger have more components, and are thus more likely to have at least one component that is not viable, and thus also not be viable.

    - You have to build further in the dark before knowing if your assemblage will be viable.

    - Smaller chunks make building an assemblage cheaper, which leads to more experimentation and discovery of good ideas.

- When things get easier creators can be more ambitious.

- If the amount of effort to create an experience is very small then you can cover tons and tons of niches.

  - How much work a thing is to write defines how much it has to be generalized to get enough market to make it viable.

  - If it's a small amount of work, it doesn't have to be generalized much to have enough of a market to make it worthwhile.

- Popular search engines had to have an army of junior employees to have the illusion of dynamic UI for search queries.

  - Look for clusters of queries that have enough scope and impact to support a small bespoke team and are underserved today, then build a bespoke, hand-crafted UI for that query cluster.

  - But now you could do that with LLMs!

- An interesting pattern: using LLMs to astroturf content in an ecosystem.

  - The challenge of an ecosystem is not so much the hill climbing of quality, it's the creation of the ecosystem to a critical mass.

  - And now you can astroturf it!

- LLM generation is slow and error prone.

  - Even if it works 95% of the time, that 5% it doesn’t is hard to predict.

  - LLMs are great at answering a specific, unique question… but then the user needs to sit there and wait while the answer unspools.

  - Some use cases get enough value from the LLM to make it worth it, but there are a lot of use cases where it wouldn’t be viable if the user had to wait a long time for an answer that could very well be wrong.

  - A successful large-scale system will use LLMs to create a lot of answers, and then cache them so in the future a similar question can be answered with a retrieval, not a generation.

  - Guess at the kinds of questions users will have, then unleash a swarm of LLMs on them in a precomputation step.

  - By the time the user asks their question, there’s already a pre-cached answer ready to go.

  - The power of LLMs, but with faster results.

- Google Search was selecting over a sea of *static* content.

  - But in AI, content can be hallucinated on demand, so maybe the Google-like position is the *generator* of that content?

- Unpacking subtexts takes patience and savviness.

  - Claude is patient and pretty savvy!

  - Claude is *great* at doing straussian reads of writing to unpack hidden subtexts.

- Optimizing for serendipity is about miracle farming.

- If you don’t constantly switch the underlying provider for a service in your system then you’ll get stuck in it.

  - You'll build up more and more functionality in the overall system that subtly assumes that it works in the one way, which will be load bearing and get increasingly heavy and hard to swap.

  - If you allow swapping out providers at that layer and support all of them it forces you to keep it flexible.

    - Allows you to keep swapping a live option.

  - However, staying flexible at that layer can be very expensive.

    - You have to constantly test the various options, and design an API to be the lowest common denominator across all providers.

  - Do you want to join yourself with that one provider, deeply commit to them, so you can go fast at that layer?

    - Or do you want to retain the option value, which will have ongoing cost *throughout* your system?

  - If it's an established service that is battle tested and has no competitors that are likely better options, committing to it is fine.

  - If it's a fast moving thing, even if there's one option that is head and shoulders ahead now, it's scary... what if alternatives catch up or surpass them?

- The real world constraints will require more complexity than you think.

  - So keep your key flow you plan simple.

  - If it starts complex and gets combinatorially more complex then you’re screwed.

  - Complexity tends to go up combinatorially as you layer in more details, more fractal surface area.

- Products are discovered, not built.

  - This is self-evident, completely obvious for games.

  - It’s less obvious for other products but still true.

- Across the user journey of trip planning, the UI you want changes, from brainstorming to starting to sketch out possibilities to locking it in.

  - Today you need to switch apps throughout that journey, and your data doesn't come with you, because each is a pocket universe.

  - What if you could have the composed app change itself and morph over the lifetime of your journey?

- “Build one to throw away” feels like it will make you go slower, but you actually go faster overall!

  - The knowhow acquisition is more useful than the code acquisition.

  - The code ties you down.

    - The more you have, the more you have a Lilliputian web of things that are hard to change holding you in place.

  - The knowhow sets you free.

    - The knowhow just tunes your intuition; the more you have, the smarter you get.

- Decentralization + top-down coherence is extremely expensive.

  - Decentralization *without* top-down coherence is cheap!

    - A flood-fill of possibility.

    - Swarms of collective intelligence.

  - You can get eventual, emergent coherence easily... the tradeoff is you can't predict what form it will take.

  - As humans we tend to prefer “certainty” (or the appearance of it) over viability.

- An app is constrained.

  - But constraints make it very clear what you can do with it.

  - Not the dizzying freedom of infinite choice.

- When you're in a branching possibility space with undo, experimentation is cheap.

  - It encourages exploration, experimentation, and discovery.

  - But most branching UIs hit you over the head with it; it consumes all x/y space.

  - All of that choice and state can be overwhelming!

    - Constantly reminding you that you’re awash in a sea of options.

  - You want a subtle reminder that you're in a branching space and can undo anything and experimentation is OK... without intimidating you with the dizzying freedom of infinite choice.

- A lot of programming today requires you to know what you want to do when you start. A blinking cursor in a textbox.

  - What if you got some boxes that you could meld and morph, like clay?

  - Everything you model in 3D starts as a cube.

- In the Github Copilot feedback loop, if you don't like the recommendation, write a bit more comments and it will generate a new thing.

  - The next thing to do to steer it is fast and obvious and natural.

  - Search is similar. Don't like the results? Amend your query and try again.

  - You don't even think about it like steering, it's the obvious, natural thing to do.

- Socio-techno problems require you to grapple with both the socio and technical.

  - At large engineering-driven companies it's possible to accidentally ignore the socio component: "If I just got the founder to tell the other team the right technical answer is X, this would be solved".

  - But in situations where you have to coordinate not only *internally* but with external parties, there’s no boss to go to that can magically fix things with the right pronouncement.

  - That makes it more obvious you need to grapple with the socio components too.

- Proprietary systems can survive only if the owner invests enough to keep it alive.

  - The more fast moving the ecosystem around it the more energy it takes for the system to stay alive and not get left behind or capsize.

  - Open systems can survive if *anyone* invests enough (individually or collectively) to keep it alive.

  - A powerful asymmetry!

- "I think what you're doing is incredibly brave"

  - "Well maybe I'm being naive and reckless because I didn't *intend* for this to be brave! Now you're giving me second thoughts..."

- A website with no links in or out is just a virtual pamphlet!

  - The whole point of a website is to be connected into the web!

- "We'll fix the hallucination problem"

  - "... Will you? That's fundamentally how it works! That's not a bug, that's the core feature!"

- Everyone would prefer to live in a world where effective medium-range, medium-scope plans are possible.

  - Everyone has at least a slight preference for this; this means that even though it’s a slight asymmetry, it’s consistent, which means the effects of it will be massive.

  - All organizations ever pretend like medium-range planning definitely works if you just try hard enough and are smart enough.

  - However the direct evidence is that except when we get extremely lucky, medium-range, medium-scope plans almost never work as planned, in all but the most boring contexts.

    - Things happen, sure, but never according to plan, and often with significant swirl and coordination overhead that wouldn’t have been necessary if a too-detailed, too-tightly-held plan didn’t exist.

    - Every time the world doesn’t fit your plan, you have to do a lot of swirl to fix it.

    - The world is fractally complex and constantly changing. Plans need to be fluid to have any hope of cohering!

- If an entity is spending all of their effort surviving they don’t have energy to create value outside of themselves.

  - When an entity is thriving they’re creating value outside themselves, lifting up and supporting others.

  - Surviving is investing energy in value creation internally to the level where the fire doesn’t go out.

  - If it goes out, the entity dies and there’s no more value it will create for itself or anyone else.

- 'One hero made it happen' is an easier story to tell.

  - The reality is almost *never* that it all came down to one particular hero.

  - But the simpler story is the one that gets shared!

  - It's easier to transmit, so it can be shared more easily and shows up and dominates compared to the more complex / nuanced / longer story.

  - Mimetic fitness is not "It's right" it's "is it right enough, and is it easy and enjoyable to transmit and receive?"

  - A story that "clicks" or "fits" in the receiver's brain (e.g. that confirms their priors, makes them feel smart for a thing they already believe) will get shared more often.

  - It will be over-represented, not because people did it on purpose, but that's just the edge on what things people bother sharing, and what kinds of things people tend to actually receive.

  - The result is that most stories are about heroes and yet very little value in the world that we see was created by heroes.

- The vast majority of value for companies is created from discretionary effort of the employees.

  - Discretionary effort is not just "an hour more work."

  - It's more connective, seed planting, creative work.

  - It's work the employee did for its own sake, so it's more likely to be creative.

  - The problems the employee picks to work on will be more likely to be good challenge, not bad challenge (e.g. bureaucratic challenge).

  - Discretionary effort is very easy to undermine, to make the relationship between employee and employer more transactional.

    - To accidentally punish discretionary effort.

    - To have that kind of work have downside for the employee but no upside.

  - How can you encourage employees to *want* to apply discretionary effort?

  - Discretionary effort cannot be forced, it can only be freely given.

    - Not unlike love.

  - Discretionary effort is far more likely to create large positive indirect effects.

    - If you had to make the value legible, it would be too hard for that work.

    - But if you don’t need to make it legible because you’re doing it for its own sake, then you can just do it.

- When you can hide from the indirect effects of your actions, you can cheat.

  - There are lots and lots of moves that create local coherence and value… but at the cost of significant externalities: indirect effects.

    - Many apparently good ideas are actually on net quite bad because of this smuggled externality.

  - If you are forced to reckon with the indirect effects of your actions, it effectively internalizes some of those externalities and forces things to balance better.

  - It forces you to operate in a way that creates more value in resonance with the broader system.

- It’s impolite to share disconfirming evidence in an organization.

  - Absorbing disconfirming evidence is what makes the thing strong.

  - But sharing disconfirming evidence is impolite.

  - In an organization, what is polite in that organization will dominate what is right.

- A situation: a person in a project that points out the negative externalities.

  - The leader of the committee comes to them: "Hey, we've decided for you to not be on the project anymore, you're just making everyone sad!"

  - Indirect effects are easy to ignore.

  - Just remove the people who point them out!

  - Indirect effects will not be proactively noticed by the coordinated collective; they're too hard to point out and get everyone to see at once.

  - Only individuals can point them out.

  - Individuals are easy to silence, intentionally or unintentionally.

- In a layoff situation, the management chain is asked to identify who on the team to layoff.

  - The manager looks at the value created by each team member, and the cost.

  - For employees where the direct value and direct cost are clear, this is easy (ish).

  - But many employees have indirect effects that influence far beyond the team’s borders (sometimes much farther).

  - The manager won’t be able to see that indirect effect, and even if they could, they’re managing their direct cost of keeping that head against a fuzzy, diffuse indirect value.

    - The manager pays the direct cost, but doesn't get the indirect value, so their balance point will be wrong.

  - So you'll over-trim things that are creating indirect value.

- The illegibility reaper.

  - In layoffs it goes around and clears out whatever the org doesn’t understand.

  - The things that are important and hard are often not legible yet.

  - So you'll reap the load bearing things in the most important spaces.

  - Employees intuitively understand this: one of the strongest motivators as an employee is to work on a thing that is legible to management.

    - Not most useful or most correct, most legible.

- Optimization can’t *create,* it can just curate and sharpen.

- Whiteboards are like spreadsheets.

  - You love yours and think it's clarifying but everyone else hates it and can't make sense of it.

- One of the reasons everyone else’s spreadsheet sucks is they didn’t create it so they don’t have the intuitive understanding of how it’s wired together and what might break when they twiddle with a given wire.

  - Whereas when you created it, you have some vague memory of how things are wired together, what modifications are safe and which ones might cause an explosion elsewhere in the sheet.

- Capitalism is great at the lower layers of Maslow's hierarchy but terrible for the higher layers, where optimization destroys meaning.

  - But we use it for both!

- Every platform emerges from a product.

- Focus not on things that were hard and now easy, but things that were impossible but are now hard.

- New laws of physics will have some things that used to be hard that are now easy and vice versa.

  - A good demo of new laws of physics is a thing that is hard elsewhere and easy in your system.

- A phrase from Dropbox culture: "Demos, not memos."

- Find things that are in the adjacent possible but people don't realize it yet.

- Misinformation succeeds not in making you believe a falsehood, but in making you not believe anything at all

  - “Everything is corrupt so you can’t trust anything,” it says, as it is the one corrupting things.

  - The side that wants more entropy has an unfair advantage. It’s easy to tear things down, an order of magnitude easier.

- It’s much easier to be mad at people you don’t know.

  - Getting to know someone is scary, it makes it harder for you to be mad at them, to blame them for a thing that has harmed you.

  - Assigning blame feels good!

    - A neat and tidy narrative to make the world make sense again.

    - A simple story to communicate to ourselves and others.

    - And also, almost certainly wrong!